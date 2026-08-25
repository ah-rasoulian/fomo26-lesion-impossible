from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .braindino_vit import SpacingAwareViT3d


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass
class ViTTaskFeatures:
    """Features produced by one full-volume backbone forward pass."""

    # Final-layer CLS token: [B, E].
    cls: torch.Tensor
    # Concatenated intermediate/final spatial maps: [B, E * L, Gd, Gh, Gw].
    feature_map: torch.Tensor
    # Individual intermediate/final maps, ordered shallow to deep.
    feature_maps: Tuple[torch.Tensor, ...]
    # Final-layer spatial map: [B, E, Gd, Gh, Gw].
    final_feature_map: torch.Tensor
    # Padded input tensor spatial shape, in the network's [H, W, D] order.
    input_shape: Tuple[int, int, int]
    # Per-subject unpadded shapes, when supplied by the collate function.
    valid_spatial_shapes: Optional[torch.Tensor]


class ViTTaskModel(nn.Module):
    """Base class that applies the pretrained ViT once to the full image."""

    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.set_backbone_trainable(not freeze_backbone)

    @property
    def backbone_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen backbone should also have dropout/stochastic depth disabled.
        if self.backbone_is_frozen:
            self.backbone.eval()
        return self

    @property
    def num_feature_levels(self) -> int:
        final_index = len(self.backbone.blocks) - 1
        return len(self.backbone.intermediate_layers) + (
            0 if final_index in self.backbone.intermediate_layers else 1
        )

    @property
    def feature_channels(self) -> int:
        return self.backbone.embed_dim * self.num_feature_levels

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def forward_feature(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        valid_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> ViTTaskFeatures:
        if x.ndim != 5:
            raise ValueError(
                f"x must be [B, C, H, W, D], got {tuple(x.shape)}."
            )

        batch_size = x.shape[0]

        # The single-subject prediction dataset may return metadata without
        # a leading batch dimension.
        if spacing.ndim == 1:
            spacing = spacing.unsqueeze(0)

        if modality is not None and modality.ndim == 1:
            modality = modality.unsqueeze(0)

        if channel_mask is not None and channel_mask.ndim == 1:
            channel_mask = channel_mask.unsqueeze(0)

        if valid_spatial_shapes is not None:
            valid_spatial_shapes = torch.as_tensor(
                valid_spatial_shapes, device=x.device, dtype=torch.long
            )
            if valid_spatial_shapes.ndim == 1:
                valid_spatial_shapes = valid_spatial_shapes.unsqueeze(0)
            if valid_spatial_shapes.shape != (batch_size, 3):
                raise ValueError(
                    "valid_spatial_shapes must be [B, 3], got "
                    f"{tuple(valid_spatial_shapes.shape)}."
                )

        if spacing.shape[0] != batch_size:
            raise ValueError(
                "spacing batch dimension must match x: "
                f"x={batch_size}, spacing={tuple(spacing.shape)}."
            )

        if modality is not None and modality.shape[0] != batch_size:
            raise ValueError(
                "modality batch dimension must match x: "
                f"x={batch_size}, modality={tuple(modality.shape)}."
            )

        if channel_mask is not None and channel_mask.shape[0] != batch_size:
            raise ValueError(
                "channel_mask batch dimension must match x: "
                f"x={batch_size}, channel_mask={tuple(channel_mask.shape)}."
            )

        if self.backbone_is_frozen:
            with torch.no_grad():
                features = self.backbone.forward_features(
                    x,
                    spacing,
                    modality,
                    channel_mask,
                    valid_spatial_shapes=valid_spatial_shapes,
                    return_feature_map=True,
                    return_intermediate=True,
                )
        else:
            features = self.backbone.forward_features(
                x,
                spacing,
                modality,
                channel_mask,
                valid_spatial_shapes=valid_spatial_shapes,
                return_feature_map=True,
                return_intermediate=True,
            )

        if features.feature_map is None:
            raise RuntimeError("Backbone did not return its final feature map.")

        levels = tuple(features.intermediate_feature_maps)
        final_index = len(self.backbone.blocks) - 1

        if final_index not in self.backbone.intermediate_layers:
            levels = (*levels, features.feature_map)

        return ViTTaskFeatures(
            cls=features.cls,
            feature_map=torch.cat(levels, dim=1),
            feature_maps=levels,
            final_feature_map=features.feature_map,
            input_shape=tuple(int(v) for v in x.shape[2:]),
            valid_spatial_shapes=valid_spatial_shapes,
        )


class _LightUpBlock(nn.Module):
    """A lightweight 3D U-Net up block with one transformer skip map."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.skip_projection = nn.Conv3d(
            skip_channels, out_channels, kernel_size=1, bias=False
        )
        merged_channels = in_channels + out_channels
        # Depthwise spatial filtering followed by a pointwise channel mix is
        # considerably lighter than two standard 3D convolutions.
        self.refine = nn.Sequential(
            nn.Conv3d(
                merged_channels,
                merged_channels,
                kernel_size=3,
                padding=1,
                groups=merged_channels,
                bias=False,
            ),
            nn.Conv3d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="trilinear",
            align_corners=False,
        )
        skip = self.skip_projection(skip)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return self.refine(torch.cat((x, skip), dim=1))


class LightUNetDecoder(nn.Module):
    """Three-stage ×2 decoder for a ViT with patch stride eight."""

    def __init__(
        self,
        embed_dim: int,
        output_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 32:
            raise ValueError("hidden_dim must be at least 32.")
        channels = (hidden_dim, max(hidden_dim // 2, 16), max(hidden_dim // 4, 8))
        self.deep_projection = nn.Sequential(
            nn.Conv3d(embed_dim, channels[0], kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.GELU(),
        )
        self.up_blocks = nn.ModuleList(
            (
                _LightUpBlock(channels[0], embed_dim, channels[0], dropout),
                _LightUpBlock(channels[0], embed_dim, channels[1], dropout),
                _LightUpBlock(channels[1], embed_dim, channels[2], dropout),
            )
        )
        self.output_head = nn.Conv3d(channels[2], output_channels, kernel_size=1)

    def forward(
        self,
        feature_maps: Tuple[torch.Tensor, ...],
        output_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        if len(feature_maps) < 4:
            raise ValueError(
                "LightUNetDecoder requires at least four feature levels, "
                f"but received {len(feature_maps)}."
            )

        # Use the four deepest maps. Earlier maps serve as progressively
        # shallower skips. Although ViT maps share the 1/8 grid initially,
        # each skip is resized to the corresponding decoder resolution.
        selected = feature_maps[-4:]
        x = self.deep_projection(selected[-1])
        for block, skip in zip(self.up_blocks, reversed(selected[:-1])):
            x = block(x, skip)

        logits = self.output_head(x)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="trilinear",
                align_corners=False,
            )
        return logits


class ViTSegModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        output_channels: int = 1,
        decoder_hidden_dim: int = 256,
        decoder_dropout: float = 0.0,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(backbone, freeze_backbone=freeze_backbone)
        self.decoder = LightUNetDecoder(
            embed_dim=backbone.embed_dim,
            output_channels=output_channels,
            hidden_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
        )

    def forward(
        self, x, spacing, modality=None, channel_mask=None,
        valid_spatial_shapes=None,
    ):
        features = self.forward_feature(
            x, spacing, modality, channel_mask, valid_spatial_shapes
        )
        return self.decoder(features.feature_maps, features.input_shape)


class _VectorHead(nn.Module):
    """A normalized two-layer prediction head for one feature vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1.")
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class _TopKClassificationHead(nn.Module):
    """Fuse global CLS evidence with top-k token-level MIL evidence."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        top_k_fraction: float = 0.05,
        min_top_k: int = 1,
        token_logit_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1].")
        if min_top_k < 1:
            raise ValueError("min_top_k must be at least 1.")
        if token_logit_weight < 0.0:
            raise ValueError("token_logit_weight must be non-negative.")

        self.top_k_fraction = float(top_k_fraction)
        self.min_top_k = int(min_top_k)
        self.token_logit_weight = float(token_logit_weight)
        self.cls_head = _VectorHead(
            embed_dim, num_classes, hidden_dim, dropout
        )
        # This MLP is shared by every spatial token.
        self.token_head = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cls: torch.Tensor,
        final_feature_map: torch.Tensor,
    ) -> torch.Tensor:
        tokens = final_feature_map.flatten(2).transpose(1, 2)  # [B, N, E]
        token_logits = self.token_head(tokens)  # [B, N, K]
        num_tokens = token_logits.shape[1]
        k = min(
            num_tokens,
            max(self.min_top_k, ceil(self.top_k_fraction * num_tokens)),
        )
        top_k_logits = token_logits.topk(k, dim=1).values.mean(dim=1)
        return self.cls_head(cls) + self.token_logit_weight * top_k_logits


class _DominantCLSRegressionHead(nn.Module):
    """Regress from CLS and mean patch features with CLS-dominant fusion."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        cls_weight: float = 0.80,
    ) -> None:
        super().__init__()
        if not 0.5 < cls_weight <= 1.0:
            raise ValueError("cls_weight must be in (0.5, 1].")
        self.cls_weight = float(cls_weight)
        self.cls_head = _VectorHead(embed_dim, output_dim, hidden_dim, dropout)
        self.patch_head = _VectorHead(embed_dim, output_dim, hidden_dim, dropout)

    def forward(
        self,
        cls: torch.Tensor,
        final_feature_map: torch.Tensor,
    ) -> torch.Tensor:
        patch_mean = final_feature_map.mean(dim=(2, 3, 4))
        cls_prediction = self.cls_head(cls)
        patch_prediction = self.patch_head(patch_mean)
        return (
            self.cls_weight * cls_prediction
            + (1.0 - self.cls_weight) * patch_prediction
        )


class ViTClsModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        top_k_fraction: float = 0.05,
        min_top_k: int = 1,
        token_logit_weight: float = 1.0,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(backbone, freeze_backbone=freeze_backbone)
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.num_classes = int(num_classes)
        self.classifier = _TopKClassificationHead(
            embed_dim=backbone.embed_dim,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            top_k_fraction=top_k_fraction,
            min_top_k=min_top_k,
            token_logit_weight=token_logit_weight,
        )

    def forward(
        self, x, spacing, modality=None, channel_mask=None,
        valid_spatial_shapes=None,
    ):
        features = self.forward_feature(
            x, spacing, modality, channel_mask, valid_spatial_shapes
        )
        return self.classifier(features.cls, features.final_feature_map)

    def predict_proba(
        self, x, spacing, modality=None, channel_mask=None,
        valid_spatial_shapes=None,
    ):
        return self.forward(
            x, spacing, modality, channel_mask, valid_spatial_shapes
        ).softmax(dim=-1)


class ViTRegModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        output_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        cls_weight: float = 0.80,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(backbone, freeze_backbone=freeze_backbone)
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1.")
        self.output_dim = int(output_dim)
        self.regressor = _DominantCLSRegressionHead(
            embed_dim=backbone.embed_dim,
            output_dim=self.output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            cls_weight=cls_weight,
        )

    def forward(
        self, x, spacing, modality=None, channel_mask=None,
        valid_spatial_shapes=None,
    ):
        features = self.forward_feature(
            x, spacing, modality, channel_mask, valid_spatial_shapes
        )
        return self.regressor(features.cls, features.final_feature_map)


def load_braindino_student_backbone(
    model: ViTTaskModel,
    checkpoint: Union[str, Path, Mapping[str, Any]],
    *,
    min_loaded_fraction: float = 0.90,
) -> nn.modules.module._IncompatibleKeys:
    """Load only the pretrained student backbone into a downstream model.

    Lightning and ``torch.compile`` wrapper prefixes are tolerated. Teacher,
    DINO/iBOT heads, centers, and all downstream task-head parameters are
    deliberately ignored.
    """
    if not 0.0 < min_loaded_fraction <= 1.0:
        raise ValueError("min_loaded_fraction must be in (0, 1].")

    payload: Any = checkpoint
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a path or a mapping.")

    state: Mapping[str, Any] = payload
    for container_key in ("state_dict", "model_state_dict"):
        candidate = state.get(container_key)
        if isinstance(candidate, Mapping):
            state = candidate
            break

    backbone_state = {}
    markers = ("student.backbone.", "student_backbone.", "backbone.")
    target_keys = set(model.backbone.state_dict())
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue
        key = raw_key.replace("_orig_mod.", "")
        mapped = None
        # Prefer an explicit student path; never fall back from a teacher key.
        if "teacher" in key.split("."):
            continue
        for marker in markers:
            position = key.find(marker)
            if position >= 0:
                mapped = key[position + len(marker):]
                break
        if mapped is None and key in target_keys:
            mapped = key
        if mapped in target_keys:
            backbone_state[mapped] = value

    loaded_fraction = len(backbone_state) / max(len(target_keys), 1)
    if loaded_fraction < min_loaded_fraction:
        raise RuntimeError(
            "Too few student-backbone tensors matched: "
            f"{len(backbone_state)}/{len(target_keys)} "
            f"({loaded_fraction:.1%}); required {min_loaded_fraction:.1%}. "
            "Check that the task architecture matches the pretraining config."
        )
    return model.backbone.load_state_dict(backbone_state, strict=False)


def _task_vit_b_backbone(
    grid_size: Union[int, Sequence[int]] = (24, 24, 24),
    patch_kernel_size: Union[int, Sequence[int]] = 8,
    window_size: Union[int, Sequence[int]] = (4, 4, 4),
    summary_grid_size: Union[int, Sequence[int]] = (8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
) -> SpacingAwareViT3d:
    """Create the ViT-B backbone used during BrainDINO pretraining."""
    return SpacingAwareViT3d(
        embed_dim=768,
        depth=12,
        num_heads=12,
        grid_size=grid_size,
        mlp_ratio=4.0,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        num_register_tokens=8,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
        position_num_bands=16,
        dropout=0.0,
        attention_dropout=0.0,
        intermediate_layers=(2, 5, 8, 11),
    )


def task_vit_b_seg(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    output_channels: int = 1,
    decoder_hidden_dim: int = 128,
    decoder_dropout: float = 0.2,
    freeze_backbone: bool = True,
) -> ViTSegModel:
    backbone = _task_vit_b_backbone(
        grid_size,
        patch_kernel_size,
        window_size,
        summary_grid_size,
        init_sigma_mm,
        learn_sigma,
    )
    return ViTSegModel(
        backbone=backbone,
        output_channels=output_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
        freeze_backbone=freeze_backbone,
    )


def task_vit_b_cls(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    num_classes: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    top_k_fraction: float = 0.05,
    min_top_k: int = 1,
    token_logit_weight: float = 1.0,
    freeze_backbone: bool = True,
) -> ViTClsModel:
    backbone = _task_vit_b_backbone(
        grid_size,
        patch_kernel_size,
        window_size,
        summary_grid_size,
        init_sigma_mm,
        learn_sigma,
    )
    return ViTClsModel(
        backbone=backbone,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        top_k_fraction=top_k_fraction,
        min_top_k=min_top_k,
        token_logit_weight=token_logit_weight,
        freeze_backbone=freeze_backbone,
    )


def task_vit_b_reg(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    output_dim: int = 1,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    cls_weight: float = 0.80,
    freeze_backbone: bool = True,
) -> ViTRegModel:
    backbone = _task_vit_b_backbone(
        grid_size,
        patch_kernel_size,
        window_size,
        summary_grid_size,
        init_sigma_mm,
        learn_sigma,
    )
    return ViTRegModel(
        backbone=backbone,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        cls_weight=cls_weight,
        freeze_backbone=freeze_backbone,
    )
