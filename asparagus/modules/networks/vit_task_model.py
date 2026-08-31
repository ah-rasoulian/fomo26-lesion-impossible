from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
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
    # Final spatial map retained under the legacy aggregate field: [B,E,G...].
    feature_map: torch.Tensor
    # Compatibility tuple containing only final_feature_map.
    feature_maps: Tuple[torch.Tensor, ...]
    # Final-layer spatial map: [B, E, Gd, Gh, Gw].
    final_feature_map: torch.Tensor
    # Padded input tensor spatial shape, in the network's [H, W, D] order.
    input_shape: Tuple[int, int, int]
    # Per-subject unpadded shapes, when supplied by the collate function.
    valid_spatial_shapes: Optional[torch.Tensor]
    original_spatial_shapes: torch.Tensor
    processed_spatial_shapes: torch.Tensor
    stride_vox: torch.Tensor
    effective_kernel_size_vox: torch.Tensor
    padding_vox: torch.Tensor
    cropping_vox: torch.Tensor
    interpolation_applied: torch.Tensor


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
        # Downstream task models intentionally consume only the final spatial
        # feature map. Intermediate transformer activations are not returned.
        return 1

    @property
    def feature_channels(self) -> int:
        return self.backbone.embed_dim * self.num_feature_levels

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def set_trainable_backbone_blocks(self, num_blocks: int) -> None:
        depth = len(self.backbone.blocks)
        if not 0 <= num_blocks <= depth:
            raise ValueError(f"num_blocks must be in [0, {depth}], got {num_blocks}.")

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        if num_blocks == 0:
            return

        for block in self.backbone.blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

        for norm in (self.backbone.patch_norm, self.backbone.global_norm):
            for parameter in norm.parameters():
                parameter.requires_grad_(True)

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
                    return_intermediate=False,
                )
        else:
            features = self.backbone.forward_features(
                x,
                spacing,
                modality,
                channel_mask,
                valid_spatial_shapes=valid_spatial_shapes,
                return_feature_map=True,
                return_intermediate=False,
            )

        if features.feature_map is None:
            raise RuntimeError("Backbone did not return its final feature map.")

        # Keep these compatibility fields, but make them final-layer-only.
        # This avoids retaining intermediate activations during fine-tuning.
        levels = (features.feature_map,)

        return ViTTaskFeatures(
            cls=features.cls,
            feature_map=torch.cat(levels, dim=1),
            feature_maps=levels,
            final_feature_map=features.feature_map,
            input_shape=tuple(int(v) for v in x.shape[2:]),
            valid_spatial_shapes=valid_spatial_shapes,
            original_spatial_shapes=features.original_spatial_shapes,
            processed_spatial_shapes=features.processed_spatial_shapes,
            stride_vox=features.stride_vox,
            effective_kernel_size_vox=features.effective_kernel_size_vox,
            padding_vox=features.padding_vox,
            cropping_vox=features.cropping_vox,
            interpolation_applied=features.interpolation_applied,
        )


class ViTLinearProbModel(ViTTaskModel):
    """Frozen BrainDINO backbone that returns its raw final CLS embedding."""

    def __init__(self, backbone: SpacingAwareViT3d) -> None:
        super().__init__(backbone=backbone, freeze_backbone=True)
        self.set_trainable_backbone_blocks(0)

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        valid_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Reuse the task-model input normalization and geometry handling, then
        # expose the final transformer CLS token without a DINO/iBOT head or
        # any additional normalization/projection.
        features = self.forward_feature(
            x=x,
            spacing=spacing,
            modality=modality,
            channel_mask=channel_mask,
            valid_spatial_shapes=valid_spatial_shapes,
        )
        return features.cls


class _DepthwiseSeparableRefine3d(nn.Module):
    """Small residual refinement using depthwise 3x3 and pointwise 1x1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if dilation < 1:
            raise ValueError("dilation must be positive.")

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.Dropout3d(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(x) + self.block(x))


class FinalFeatureSegDecoder(nn.Module):
    """Geometry-aware decoder using only the final transformer feature map.

    Expensive feature convolutions are capped at ``max_scale`` times the token
    grid. For a 24^3 token grid and the default scale of four, the largest
    decoder feature grid is 96^3. Only the low-channel output logits are then
    interpolated to each subject's native spatial shape.
    """

    def __init__(
        self,
        embed_dim: int,
        output_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        max_scale: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim < 32:
            raise ValueError("hidden_dim must be at least 32.")
        if max_scale < 2:
            raise ValueError("max_scale must be at least 2.")

        token_channels = min(int(hidden_dim), 96)
        middle_channels = max(token_channels // 2, 24)
        fine_channels = max(token_channels // 4, 16)
        self.max_scale = int(max_scale)

        self.input_projection = nn.Sequential(
            nn.Conv3d(embed_dim, token_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(token_channels), token_channels),
            nn.GELU(),
        )
        # Dilation two adds context at 24^3 without a large kernel.
        self.token_context = _DepthwiseSeparableRefine3d(
            token_channels,
            token_channels,
            dropout=dropout,
            dilation=2,
        )
        self.middle_refine = _DepthwiseSeparableRefine3d(
            token_channels,
            middle_channels,
            dropout=dropout,
        )
        self.fine_refine = _DepthwiseSeparableRefine3d(
            middle_channels,
            fine_channels,
            dropout=dropout,
        )
        self.output_head = nn.Conv3d(
            fine_channels,
            output_channels,
            kernel_size=1,
        )

    @staticmethod
    def _validate_geometry_tensor(
        name: str,
        value: torch.Tensor,
        expected_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.long)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got "
                f"{tuple(value.shape)}."
            )
        return value

    def _decoder_sizes(
        self,
        token_shape: Tuple[int, int, int],
        processed_shape: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        middle = tuple(
            min(processed, token * 2)
            for token, processed in zip(token_shape, processed_shape)
        )
        fine = tuple(
            min(processed, token * self.max_scale)
            for token, processed in zip(token_shape, processed_shape)
        )
        return middle, fine

    @staticmethod
    def _token_axis_coordinates(
        output_length: int,
        processed_length: int,
        token_length: int,
        stride: int,
        kernel: int,
        padding_left: int,
        cropping_left: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        output_index = torch.arange(
            output_length,
            device=device,
            dtype=torch.float32,
        )
        processed_coordinate = (
            (output_index + 0.5)
            * (float(processed_length) / float(output_length))
            - 0.5
        )
        adjusted_coordinate = (
            processed_coordinate
            - float(cropping_left)
            + float(padding_left)
        )
        first_token_center = (float(kernel) - 1.0) / 2.0
        token_coordinate = (
            adjusted_coordinate - first_token_center
        ) / float(stride)
        if token_length == 1:
            return torch.zeros_like(token_coordinate)
        return (
            2.0 * token_coordinate / float(token_length - 1) - 1.0
        )

    @classmethod
    def _tokens_to_processed_grid(
        cls,
        feature_map: torch.Tensor,
        output_size: Tuple[int, int, int],
        processed_shape: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        kernel: Tuple[int, int, int],
        padding: torch.Tensor,
        cropping: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map.shape[0] != 1:
            raise ValueError("Geometry resampling expects one subject.")

        token_shape = tuple(int(value) for value in feature_map.shape[2:])
        axes = tuple(
            cls._token_axis_coordinates(
                output_length=output_size[axis],
                processed_length=processed_shape[axis],
                token_length=token_shape[axis],
                stride=stride[axis],
                kernel=kernel[axis],
                padding_left=int(padding[axis, 0]),
                cropping_left=int(cropping[axis, 0]),
                device=feature_map.device,
            )
            for axis in range(3)
        )
        height, width, depth = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack((depth, width, height), dim=-1).unsqueeze(0)
        return F.grid_sample(
            feature_map,
            grid.to(dtype=feature_map.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _decode_subject(
        self,
        final_feature_map: torch.Tensor,
        original_shape: Tuple[int, int, int],
        processed_shape: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        kernel: Tuple[int, int, int],
        padding: torch.Tensor,
        cropping: torch.Tensor,
    ) -> torch.Tensor:
        token_shape = tuple(
            int(value) for value in final_feature_map.shape[2:]
        )
        middle_size, fine_size = self._decoder_sizes(
            token_shape,
            processed_shape,
        )

        x = self.input_projection(final_feature_map)
        x = self.token_context(x)
        x = self._tokens_to_processed_grid(
            x,
            middle_size,
            processed_shape,
            stride,
            kernel,
            padding,
            cropping,
        )
        x = self.middle_refine(x)

        if tuple(x.shape[2:]) != fine_size:
            x = F.interpolate(
                x,
                size=fine_size,
                mode="trilinear",
                align_corners=False,
            )
        x = self.fine_refine(x)
        logits = self.output_head(x)

        # Only output_channels values are expanded to the native image. No
        # high-channel convolution is performed at this large resolution.
        if tuple(logits.shape[2:]) != original_shape:
            logits = F.interpolate(
                logits,
                size=original_shape,
                mode="trilinear",
                align_corners=False,
            )
        return logits

    def forward(
        self,
        final_feature_map: torch.Tensor,
        output_size: Tuple[int, int, int],
        original_spatial_shapes: torch.Tensor,
        processed_spatial_shapes: torch.Tensor,
        stride_vox: torch.Tensor,
        effective_kernel_size_vox: torch.Tensor,
        padding_vox: torch.Tensor,
        cropping_vox: torch.Tensor,
    ) -> torch.Tensor:
        if final_feature_map.ndim != 5:
            raise ValueError(
                "final_feature_map must be [B, C, H, W, D], got "
                f"{tuple(final_feature_map.shape)}."
            )

        batch_size = final_feature_map.shape[0]
        original_spatial_shapes = self._validate_geometry_tensor(
            "original_spatial_shapes",
            original_spatial_shapes,
            (batch_size, 3),
        )
        processed_spatial_shapes = self._validate_geometry_tensor(
            "processed_spatial_shapes",
            processed_spatial_shapes,
            (batch_size, 3),
        )
        stride_vox = self._validate_geometry_tensor(
            "stride_vox",
            stride_vox,
            (batch_size, 3),
        )
        effective_kernel_size_vox = self._validate_geometry_tensor(
            "effective_kernel_size_vox",
            effective_kernel_size_vox,
            (batch_size, 3),
        )
        padding_vox = self._validate_geometry_tensor(
            "padding_vox",
            padding_vox,
            (batch_size, 3, 2),
        )
        cropping_vox = self._validate_geometry_tensor(
            "cropping_vox",
            cropping_vox,
            (batch_size, 3, 2),
        )

        padded_logits = []
        for index in range(batch_size):
            original_shape = tuple(
                int(value) for value in original_spatial_shapes[index]
            )
            processed_shape = tuple(
                int(value) for value in processed_spatial_shapes[index]
            )
            if any(
                original > canvas
                for original, canvas in zip(original_shape, output_size)
            ):
                raise ValueError(
                    f"Subject {index} shape {original_shape} exceeds "
                    f"the output canvas {output_size}."
                )

            subject_logits = self._decode_subject(
                final_feature_map[index : index + 1],
                original_shape=original_shape,
                processed_shape=processed_shape,
                stride=tuple(int(value) for value in stride_vox[index]),
                kernel=tuple(
                    int(value)
                    for value in effective_kernel_size_vox[index]
                ),
                padding=padding_vox[index],
                cropping=cropping_vox[index],
            )
            pad_height = output_size[0] - original_shape[0]
            pad_width = output_size[1] - original_shape[1]
            pad_depth = output_size[2] - original_shape[2]
            subject_logits = F.pad(
                subject_logits,
                (0, pad_depth, 0, pad_width, 0, pad_height),
                mode="constant",
                value=0.0,
            )
            padded_logits.append(subject_logits)

        return torch.cat(padded_logits, dim=0)


class ViTSegModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        output_channels: int = 2,
        decoder_hidden_dim: int = 64,
        decoder_dropout: float = 0.1,
        decoder_max_scale: int = 4,
        initial_foreground_probability: float = 0.10,
        freeze_backbone: bool = True,
        trainable_backbone_blocks: int = 0,
    ) -> None:
        super().__init__(backbone=backbone, freeze_backbone=freeze_backbone)
        if output_channels < 2:
            raise ValueError(
                "output_channels must include background and be at least 2."
            )
        if trainable_backbone_blocks < 0:
            raise ValueError("trainable_backbone_blocks must be non-negative.")
        if trainable_backbone_blocks > len(backbone.blocks):
            raise ValueError(
                f"Backbone has {len(backbone.blocks)} blocks, but "
                f"{trainable_backbone_blocks} were requested."
            )
        if trainable_backbone_blocks > 0:
            self.set_trainable_backbone_blocks(trainable_backbone_blocks)
        elif freeze_backbone:
            self.set_trainable_backbone_blocks(0)

        # These pretrained parameters do not participate in the segmentation
        # loss. Freezing them also keeps standard DDP from reporting unused
        # trainable parameters.
        self.backbone.mask_token.requires_grad_(False)
        for parameter in self.backbone.global_norm.parameters():
            parameter.requires_grad_(False)

        self.output_channels = int(output_channels)
        self.decoder = FinalFeatureSegDecoder(
            embed_dim=backbone.embed_dim,
            output_channels=self.output_channels,
            hidden_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
            max_scale=decoder_max_scale,
        )
        self._initialize_segmentation_output(initial_foreground_probability)

    def _initialize_segmentation_output(
        self,
        foreground_probability: float,
    ) -> None:
        """Initialize softmax priors without blocking decoder gradients.

        For the binary Task 2 configuration, values above 0.5 make the first
        hard prediction foreground. A very small nonzero kernel initialization
        keeps gradients flowing into the rest of the decoder on step one.
        """
        probability = float(foreground_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError(
                "initial_foreground_probability must be in (0, 1)."
            )

        output_head = self.decoder.output_head
        nn.init.normal_(output_head.weight, mean=0.0, std=1e-3)
        background_probability = 1.0 - probability
        per_foreground_probability = probability / (self.output_channels - 1)
        with torch.no_grad():
            output_head.bias[0] = log(background_probability)
            output_head.bias[1:] = log(per_foreground_probability)

    def forward(
        self,
        x,
        spacing,
        modality=None,
        channel_mask=None,
        valid_spatial_shapes=None,
    ):
        features = self.forward_feature(
            x=x,
            spacing=spacing,
            modality=modality,
            channel_mask=channel_mask,
            valid_spatial_shapes=valid_spatial_shapes,
        )
        return self.decoder(
            final_feature_map=features.final_feature_map,
            output_size=features.input_shape,
            original_spatial_shapes=features.original_spatial_shapes,
            processed_spatial_shapes=features.processed_spatial_shapes,
            stride_vox=features.stride_vox,
            effective_kernel_size_vox=features.effective_kernel_size_vox,
            padding_vox=features.padding_vox,
            cropping_vox=features.cropping_vox,
        )


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
        trainable_backbone_blocks: int = 0,
    ) -> None:
        super().__init__(backbone, freeze_backbone=True)
        self.set_trainable_backbone_blocks(trainable_backbone_blocks)
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
        trainable_backbone_blocks: int = 0,
    ) -> None:
        super().__init__(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
        )

        if output_dim < 1:
            raise ValueError("output_dim must be at least 1.")

        if trainable_backbone_blocks < 0:
            raise ValueError(
                "trainable_backbone_blocks must be non-negative."
            )

        self.output_dim = int(output_dim)

        # This overrides the initial freeze_backbone setting when partial
        # backbone fine-tuning is requested.
        if trainable_backbone_blocks > 0:
            self.set_trainable_backbone_blocks(
                trainable_backbone_blocks
            )
        elif freeze_backbone:
            self.set_trainable_backbone_blocks(0)

        self.regressor = _DominantCLSRegressionHead(
            embed_dim=backbone.embed_dim,
            output_dim=self.output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            cls_weight=cls_weight,
        )

    def forward(
        self,
        x,
        spacing,
        modality=None,
        channel_mask=None,
        valid_spatial_shapes=None,
    ):
        features = self.forward_feature(
            x=x,
            spacing=spacing,
            modality=modality,
            channel_mask=channel_mask,
            valid_spatial_shapes=valid_spatial_shapes,
        )
        return self.regressor(
            features.cls,
            features.final_feature_map,
        )


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

def _task_vit_s_backbone(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
) -> SpacingAwareViT3d:
    return SpacingAwareViT3d(
        embed_dim=384,
        depth=12,
        num_heads=6,
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
def task_vit_s_seg(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    output_channels: int = 2,
    decoder_hidden_dim: int = 64,
    decoder_dropout: float = 0.1,
    decoder_max_scale: int = 4,
    initial_foreground_probability: float = 0.10,
    freeze_backbone: bool = True,
    trainable_backbone_blocks: int = 0,
) -> ViTSegModel:
    backbone = _task_vit_s_backbone(
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )
    return ViTSegModel(
        backbone=backbone,
        output_channels=output_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
        decoder_max_scale=decoder_max_scale,
        initial_foreground_probability=initial_foreground_probability,
        freeze_backbone=freeze_backbone,
        trainable_backbone_blocks=trainable_backbone_blocks,
    )


def task_vit_b_seg(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    output_channels: int = 2,
    decoder_hidden_dim: int = 64,
    decoder_dropout: float = 0.1,
    decoder_max_scale: int = 4,
    initial_foreground_probability: float = 0.10,
    freeze_backbone: bool = True,
    trainable_backbone_blocks: int = 0,
) -> ViTSegModel:
    backbone = _task_vit_b_backbone(
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )
    return ViTSegModel(
        backbone=backbone,
        output_channels=output_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
        decoder_max_scale=decoder_max_scale,
        initial_foreground_probability=initial_foreground_probability,
        freeze_backbone=freeze_backbone,
        trainable_backbone_blocks=trainable_backbone_blocks,
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
    trainable_backbone_blocks: int = 0,
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
        trainable_backbone_blocks=trainable_backbone_blocks,
    )

def task_vit_s_cls(
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
    trainable_backbone_blocks: int = 0,
) -> ViTClsModel:
    backbone = _task_vit_s_backbone(
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
        trainable_backbone_blocks=trainable_backbone_blocks,
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
    trainable_backbone_blocks: int = 0,
) -> ViTRegModel:
    backbone = _task_vit_b_backbone(
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )

    return ViTRegModel(
        backbone=backbone,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        cls_weight=cls_weight,
        freeze_backbone=freeze_backbone,
        trainable_backbone_blocks=trainable_backbone_blocks,
    )

def task_vit_s_reg(
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
    trainable_backbone_blocks: int = 0,
) -> ViTRegModel:
    backbone = _task_vit_s_backbone(
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )

    return ViTRegModel(
        backbone=backbone,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        cls_weight=cls_weight,
        freeze_backbone=freeze_backbone,
        trainable_backbone_blocks=trainable_backbone_blocks,
    )


def task_vit_b_linear_prob(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
) -> ViTLinearProbModel:
    backbone = _task_vit_b_backbone(
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )
    return ViTLinearProbModel(backbone=backbone)


def task_vit_s_linear_prob(
    grid_size=(24, 24, 24),
    patch_kernel_size=8,
    window_size=(4, 4, 4),
    summary_grid_size=(8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
) -> ViTLinearProbModel:
    backbone = _task_vit_s_backbone(
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
    )
    return ViTLinearProbModel(backbone=backbone)
