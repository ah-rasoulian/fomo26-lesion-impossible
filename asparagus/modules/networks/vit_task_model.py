from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spacing_aware_vit import SpacingAwareViT3d


@dataclass
class ViTTaskFeatures:
    """Features produced by one full-volume backbone forward pass."""

    # Final-layer CLS token: [B, E].
    cls: torch.Tensor
    # Concatenated intermediate/final spatial maps: [B, E * L, Gd, Gh, Gw].
    feature_map: torch.Tensor
    # Final-layer spatial map: [B, E, Gd, Gh, Gw].
    final_feature_map: torch.Tensor
    # Original full-volume spatial shape.
    input_shape: Tuple[int, int, int]


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
    ) -> ViTTaskFeatures:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, D, H, W], got {tuple(x.shape)}.")

        if self.backbone_is_frozen:
            with torch.no_grad():
                features = self.backbone.forward_features(
                    x, spacing, modality, channel_mask
                )
        else:
            features = self.backbone.forward_features(
                x, spacing, modality, channel_mask
            )

        levels = tuple(features.intermediate_feature_maps)
        final_index = len(self.backbone.blocks) - 1
        if final_index not in self.backbone.intermediate_layers:
            levels = (*levels, features.feature_map)

        return ViTTaskFeatures(
            cls=features.cls,
            feature_map=torch.cat(levels, dim=1),
            final_feature_map=features.feature_map,
            input_shape=tuple(int(v) for v in x.shape[2:]),
        )


class MLPDecoder(nn.Module):
    """Apply the same small MLP to every spatial token."""

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        # Conv1d with kernel size 1 is a shared MLP over flattened tokens.
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, output_channels, kernel_size=1),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        output_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, _, grid_d, grid_h, grid_w = feature_map.shape
        tokens = feature_map.flatten(2)
        logits = self.mlp(tokens)
        logits = logits.reshape(batch_size, -1, grid_d, grid_h, grid_w)
        return F.interpolate(
            logits,
            size=output_size,
            mode="trilinear",
            align_corners=False,
        )


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
        self.decoder = MLPDecoder(
            in_channels=self.feature_channels,
            output_channels=output_channels,
            hidden_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
        )

    def forward(self, x, spacing, modality=None, channel_mask=None):
        features = self.forward_feature(x, spacing, modality, channel_mask)
        return self.decoder(features.feature_map, features.input_shape)


class _CLSTokenHead(nn.Module):
    """Small MLP operating only on the final CLS token."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.head(cls)


class ViTClsModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(backbone, freeze_backbone=freeze_backbone)
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.num_classes = int(num_classes)
        self.classifier = _CLSTokenHead(
            backbone.embed_dim, self.num_classes, hidden_dim, dropout
        )

    def forward(self, x, spacing, modality=None, channel_mask=None):
        features = self.forward_feature(x, spacing, modality, channel_mask)
        return self.classifier(features.cls)

    def predict_proba(self, x, spacing, modality=None, channel_mask=None):
        return self.forward(x, spacing, modality, channel_mask).softmax(dim=-1)


class ViTRegModel(ViTTaskModel):
    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        output_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(backbone, freeze_backbone=freeze_backbone)
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1.")
        self.output_dim = int(output_dim)
        self.regressor = _CLSTokenHead(
            backbone.embed_dim, self.output_dim, hidden_dim, dropout
        )

    def forward(self, x, spacing, modality=None, channel_mask=None):
        features = self.forward_feature(x, spacing, modality, channel_mask)
        return self.regressor(features.cls)


def _task_vit_b_backbone(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,
) -> SpacingAwareViT3d:
    """Create the ViT-B backbone used during BrainDINO pretraining."""
    return SpacingAwareViT3d(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
        position_num_bands=16,
        dropout=0.0,
        attention_dropout=0.0,
        intermediate_layers=(2, 5, 8, 11),
    )


def task_vit_b_seg(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,
    output_channels: int = 1,
    decoder_hidden_dim: int = 128,
    decoder_dropout: float = 0.20,
    freeze_backbone: bool = True,
) -> ViTSegModel:
    backbone = _task_vit_b_backbone(
        patch_kernel_size,
        patch_stride,
        patch_padding,
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
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,
    num_classes: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.20,
    freeze_backbone: bool = True,
) -> ViTClsModel:
    backbone = _task_vit_b_backbone(
        patch_kernel_size,
        patch_stride,
        patch_padding,
        init_sigma_mm,
        learn_sigma,
    )
    return ViTClsModel(
        backbone=backbone,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def task_vit_b_reg(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,
    output_dim: int = 1,
    hidden_dim: int = 128,
    dropout: float = 0.20,
    freeze_backbone: bool = True,
) -> ViTRegModel:
    backbone = _task_vit_b_backbone(
        patch_kernel_size,
        patch_stride,
        patch_padding,
        init_sigma_mm,
        learn_sigma,
    )
    return ViTRegModel(
        backbone=backbone,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )