from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spacing_aware_vit import SpacingAwareViT3d


Int3 = Union[int, Sequence[int]]


def _triple(value: Int3) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    if len(value) != 3:
        raise ValueError(f"Expected three values, got {value}.")
    return tuple(int(v) for v in value)


@dataclass
class ViTTaskFeatures:
    """Whole-image representation assembled from sliding-window crops."""

    # Attention-pooled crop CLS tokens: [1, E].
    cls: torch.Tensor
    # Concatenated final and intermediate spatial maps: [1, E * L, Gh, Gw, Gd].
    feature_map: torch.Tensor
    # Final-layer map only: [1, E, Gh, Gw, Gd].
    final_feature_map: torch.Tensor
    # Individual CLS tokens and their normalized aggregation weights.
    window_cls: torch.Tensor
    cls_weights: torch.Tensor
    # Input-space crop starts and the unpadded input shape.
    window_starts: Tuple[Tuple[int, int, int], ...]
    input_shape: Tuple[int, int, int]


class GatedCLSAggregator(nn.Module):
    """Learned multiple-instance pooling over crop-level CLS tokens."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        hidden_dim = min(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [number of windows, E]
        scores = self.score(self.norm(tokens)).squeeze(-1)
        if validity is not None:
            # A fixed prior prevents heavily padded edge crops from receiving
            # disproportionate attention before the small-data head is trained.
            scores = scores + validity.clamp_min(1e-6).log()
        weights = scores.softmax(dim=0)
        pooled = torch.sum(tokens * weights[:, None], dim=0, keepdim=True)
        return pooled, weights


class ViTTaskModel(nn.Module):
    """
    Parent model for whole-volume downstream inference/fine-tuning.

    The input batch size is deliberately restricted to one. Windows are
    processed sequentially to keep memory bounded. Spatial tokens from
    overlapping windows are blended with a Gaussian importance map. The final
    and all intermediate feature maps are concatenated channel-wise, while CLS
    tokens are pooled separately with gated attention.

    Input convention: image [1, C, H, W, D], spacing [3] or [1, 3], and
    modality/channel_mask [1, C]. Sliding-window starts are aligned to the
    backbone token stride so token grids can be stitched without interpolation.
    """

    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        window_size: Int3 = (128, 128, 128),
        window_overlap: float = 0.5,
        cls_attention_dim: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= window_overlap < 1.0:
            raise ValueError("window_overlap must be in [0, 1).")
        self.backbone = backbone
        self.window_size = _triple(window_size)
        self.window_overlap = float(window_overlap)
        self.cls_aggregator = GatedCLSAggregator(
            backbone.embed_dim, cls_attention_dim
        )
        self.set_backbone_trainable(not freeze_backbone)

    @property
    def num_feature_levels(self) -> int:
        has_final = (len(self.backbone.blocks) - 1) in self.backbone.intermediate_layers
        return len(self.backbone.intermediate_layers) + (0 if has_final else 1)

    @property
    def feature_channels(self) -> int:
        return self.backbone.embed_dim * self.num_feature_levels

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def _axis_starts(self, padded_size: int, window: int, token_stride: int):
        if padded_size <= window:
            return [0]
        desired = max(token_stride, int(round(window * (1.0 - self.window_overlap))))
        step = max(token_stride, (desired // token_stride) * token_stride)
        starts = list(range(0, padded_size - window + 1, step))
        last = padded_size - window
        # Padding computed below makes this aligned, but retain the assertion to
        # protect stitching if the implementation is changed later.
        if starts[-1] != last:
            starts.append(last)
        if any(start % token_stride for start in starts):
            raise RuntimeError("Sliding-window starts must align to patch stride.")
        return starts

    def _padded_axis_size(self, size: int, window: int, token_stride: int) -> int:
        if size <= window:
            return window
        desired = max(token_stride, int(round(window * (1.0 - self.window_overlap))))
        step = max(token_stride, (desired // token_stride) * token_stride)
        steps = (size - window + step - 1) // step
        return window + steps * step

    @staticmethod
    def _importance_map(
        shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        axes = [
            torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
            for n in shape
        ]
        grid = torch.meshgrid(*axes, indexing="ij")
        distance = sum(axis.square() for axis in grid)
        weight = torch.exp(-0.5 * distance / (0.5 ** 2)).clamp_min(1e-3)
        return weight.to(dtype=dtype)[None, None]

    def forward_feature(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> ViTTaskFeatures:
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(
                f"x must be [1, C, H, W, D]; got {tuple(x.shape)}."
            )
        input_shape = tuple(int(v) for v in x.shape[2:])
        token_stride = tuple(int(v) for v in self.backbone.patch_embed.patch_stride)
        padded_shape = tuple(
            self._padded_axis_size(size, window, stride)
            for size, window, stride in zip(
                input_shape, self.window_size, token_stride
            )
        )
        pad = tuple(
            value
            for original, padded in reversed(tuple(zip(input_shape, padded_shape)))
            for value in (0, padded - original)
        )
        x = F.pad(x, pad, mode="constant", value=0.0)

        axis_starts = [
            self._axis_starts(size, window, stride)
            for size, window, stride in zip(
                padded_shape, self.window_size, token_stride
            )
        ]
        starts = tuple(product(*axis_starts))

        accumulators = None
        weight_sum = None
        cls_tokens = []
        window_validity = []

        for start in starts:
            slices = tuple(
                slice(origin, origin + size)
                for origin, size in zip(start, self.window_size)
            )
            crop = x[(slice(None), slice(None), *slices)]
            features = self.backbone.forward_features(
                crop, spacing, modality, channel_mask
            )
            levels = features.intermediate_feature_maps
            # Always make the true final map the last level, without duplicating
            # it when the final transformer block is an intermediate tap.
            if (len(self.backbone.blocks) - 1) not in self.backbone.intermediate_layers:
                levels = (*levels, features.feature_map)

            if accumulators is None:
                local_grid = tuple(int(v) for v in levels[0].shape[2:])
                offsets = tuple(s // t for s, t in zip(start, token_stride))
                global_grid = tuple(
                    o + g for o, g in zip(
                        tuple(
                            (padded - window) // stride
                            for padded, window, stride in zip(
                                padded_shape, self.window_size, token_stride
                            )
                        ),
                        local_grid,
                    )
                )
                importance = self._importance_map(
                    local_grid, levels[0].device, levels[0].dtype
                )
                accumulators = [
                    level.new_zeros((1, level.shape[1], *global_grid))
                    for level in levels
                ]
                weight_sum = levels[0].new_zeros((1, 1, *global_grid))

            offsets = tuple(s // t for s, t in zip(start, token_stride))
            target = tuple(
                slice(offset, offset + extent)
                for offset, extent in zip(offsets, local_grid)
            )
            for accumulator, level in zip(accumulators, levels):
                accumulator[(slice(None), slice(None), *target)] += level * importance
            weight_sum[(slice(None), slice(None), *target)] += importance
            cls_tokens.append(features.cls.squeeze(0))
            valid_voxels = 1
            crop_voxels = 1
            for origin, window, original in zip(
                start, self.window_size, input_shape
            ):
                valid_voxels *= max(0, min(origin + window, original) - origin)
                crop_voxels *= window
            window_validity.append(valid_voxels / crop_voxels)

        stitched_levels = tuple(
            accumulator / weight_sum.clamp_min(1e-6)
            for accumulator in accumulators
        )
        # Remove token positions that correspond only to end padding. This also
        # ensures that decoder upsampling preserves the original image geometry.
        valid_grid = tuple(
            min(size, (original + stride - 1) // stride)
            for size, original, stride in zip(
                stitched_levels[0].shape[2:], input_shape, token_stride
            )
        )
        valid_slices = tuple(slice(0, size) for size in valid_grid)
        stitched_levels = tuple(
            level[(slice(None), slice(None), *valid_slices)]
            for level in stitched_levels
        )
        window_cls = torch.stack(cls_tokens, dim=0)
        validity = window_cls.new_tensor(window_validity)
        image_cls, cls_weights = self.cls_aggregator(window_cls, validity)
        return ViTTaskFeatures(
            cls=image_cls,
            feature_map=torch.cat(stitched_levels, dim=1),
            final_feature_map=stitched_levels[-1],
            window_cls=window_cls,
            cls_weights=cls_weights,
            window_starts=starts,
            input_shape=input_shape,
        )


class LightweightSegDecoder(nn.Module):
    """Parameter-efficient token-map decoder for very small datasets."""

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int = 64
    ) -> None:
        super().__init__()
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.decode = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int, int]):
        x = self.decode(x)
        return F.interpolate(
            x, size=output_size, mode="trilinear", align_corners=False
        )


class LightweightUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        return self.block(x)


class LightweightViTUNetDecoder(nn.Module):
    """
    Progressive ×8 decoder for a ViT with token stride 8.

    Input:
        [B, in_channels, H/8, W/8, D/8]

    Output:
        [B, output_channels, H, W, D]
    """

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        decoder_channels: Sequence[int] = (64, 48, 32, 16),
    ) -> None:
        super().__init__()

        if len(decoder_channels) != 4:
            raise ValueError(
                "decoder_channels must contain four channel sizes."
            )

        c0, c1, c2, c3 = decoder_channels

        self.input_projection = nn.Sequential(
            nn.Conv3d(
                in_channels,
                c0,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, c0),
                num_channels=c0,
            ),
            nn.GELU(),
        )

        self.up_4 = LightweightUpBlock(c0, c1)
        self.up_2 = LightweightUpBlock(c1, c2)
        self.up_1 = LightweightUpBlock(c2, c3)

        self.output_head = nn.Conv3d(
            c3,
            output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        output_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        x = self.input_projection(feature_map)

        x = self.up_4(x)
        x = self.up_2(x)
        x = self.up_1(x)

        # Handles dimensions that are not exact multiples of eight.
        if x.shape[2:] != output_size:
            x = F.interpolate(
                x,
                size=output_size,
                mode="trilinear",
                align_corners=False,
            )

        return self.output_head(x)

class ViTSegModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        output_channels: int = 1,
        decoder_channels: Sequence[int] = (64, 48, 32, 16),
        **kwargs,
    ) -> None:
        super().__init__(backbone, **kwargs)
        self.decoder = LightweightViTUNetDecoder(
            in_channels=self.feature_channels,
            output_channels=output_channels,
            decoder_channels=decoder_channels,
        )

    def forward(self, x, spacing, modality=None, channel_mask=None):
        features = self.forward_feature(x, spacing, modality, channel_mask)
        return self.decoder(features.feature_map, features.input_shape)


class _GlobalTaskHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        # Combine separately pooled CLS and spatial evidence.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, output_dim),
        )

    def forward(self, cls: torch.Tensor, feature_map: torch.Tensor):
        spatial = feature_map.mean(dim=(2, 3, 4))
        return self.head(torch.cat((cls, spatial), dim=1))


class ViTClsModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        num_classes: int = 1,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(backbone, **kwargs)
        self.num_classes = num_classes
        self.classifier = _GlobalTaskHead(
            backbone.embed_dim, num_classes, dropout
        )

    def forward(self, x, spacing, modality=None, channel_mask=None):
        features = self.forward_feature(x, spacing, modality, channel_mask)
        # Return logits. For binary inference: torch.sigmoid(logits).
        return self.classifier(features.cls, features.final_feature_map)

    def predict_proba(self, x, spacing, modality=None, channel_mask=None):
        logits = self.forward(x, spacing, modality, channel_mask)
        return logits.sigmoid() if logits.shape[-1] == 1 else logits.softmax(-1)


class ViTRegModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        output_dim: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(backbone, **kwargs)
        self.output_dim = output_dim
        self.regressor = _GlobalTaskHead(backbone.embed_dim, output_dim, dropout)

    def forward(self, x, spacing, modality=None, channel_mask=None):
        features = self.forward_feature(x, spacing, modality, channel_mask)
        return self.regressor(features.cls, features.final_feature_map)


def _task_vit_b_backbone(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,
) -> SpacingAwareViT3d:
    """
    Creates the same ViT-B backbone used during BrainDINO pretraining.

    The returned backbone does not include the DINO/iBOT projection heads.
    """
    return SpacingAwareViT3d(
        # ViT-B architecture
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,

        # Spacing-aware patch embedding
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,

        # Physical Fourier position encoding
        position_num_bands=16,

        # Keep consistent with pretraining
        dropout=0.0,
        attention_dropout=0.0,

        # Transformer features used by the segmentation decoder
        intermediate_layers=(2, 5, 8, 11),
    )

def task_vit_b_seg(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,

    output_channels: int = 1,

    window_size: Int3 = (128, 128, 128),
    window_overlap: float = 0.5,

    decoder_channels: Sequence[int] = (64, 48, 32, 16),
    cls_attention_dim: int = 128,
    freeze_backbone: bool = True,
) -> ViTSegModel:
    backbone = _task_vit_b_backbone(
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )

    return ViTSegModel(
        backbone=backbone,
        output_channels=output_channels,
        decoder_channels=decoder_channels,

        window_size=window_size,
        window_overlap=window_overlap,
        cls_attention_dim=cls_attention_dim,
        freeze_backbone=freeze_backbone,
    )

def task_vit_b_cls(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,

    num_classes: int = 1,

    window_size: Int3 = (128, 128, 128),
    window_overlap: float = 0.5,

    cls_attention_dim: int = 128,
    dropout: float = 0.2,
    freeze_backbone: bool = True,
) -> ViTClsModel:
    backbone = _task_vit_b_backbone(
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )

    return ViTClsModel(
        backbone=backbone,
        num_classes=num_classes,
        dropout=dropout,

        window_size=window_size,
        window_overlap=window_overlap,
        cls_attention_dim=cls_attention_dim,
        freeze_backbone=freeze_backbone,
    )

def task_vit_b_reg(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,

    output_dim: int = 1,

    window_size: Int3 = (128, 128, 128),
    window_overlap: float = 0.5,

    cls_attention_dim: int = 128,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
) -> ViTRegModel:
    backbone = _task_vit_b_backbone(
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )

    return ViTRegModel(
        backbone=backbone,
        output_dim=output_dim,
        dropout=dropout,

        window_size=window_size,
        window_overlap=window_overlap,
        cls_attention_dim=cls_attention_dim,
        freeze_backbone=freeze_backbone,
    )
