from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from asparagus.modules.networks.blocks.layers.physical_conv3d import Int3, PhysicalGaussianMaskedConv3d, expand_spacing, to_3tuple
from asparagus.functional.loading import MODALITY_TO_ID


PretrainingMode = Literal["student", "teacher", "features"]


def _nonnegative_3tuple(value: Int3) -> Tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        if len(value) != 3:
            raise ValueError(f"Expected three values, got {value}.")
        result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"Values must be non-negative, got {result}.")
    return result


@dataclass
class PatchEmbedOutput:
    tokens: torch.Tensor
    grid_shape: Tuple[int, int, int]
    grid_spacing_mm: torch.Tensor


@dataclass
class ViTFeatures:
    cls: torch.Tensor
    registers: torch.Tensor
    patches: torch.Tensor
    patch_grid_shape: Tuple[int, int, int]
    grid_spacing_mm: torch.Tensor
    feature_map: Optional[torch.Tensor]
    intermediate_feature_maps: Tuple[torch.Tensor, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "cls_features": self.cls,
            "register_features": self.registers,
            "patch_features": self.patches,
            "patch_grid_shape": self.patch_grid_shape,
            "grid_spacing_mm": self.grid_spacing_mm,
            "feature_map": self.feature_map,
            "intermediate_feature_maps": self.intermediate_feature_maps,
        }


class PhysicalFourierPositionEmbedding3d(nn.Module):
    """Continuous Fourier positions evaluated at token centers in millimetres."""

    def __init__(
        self,
        embed_dim: int,
        num_bands: int = 16,
        min_wavelength_mm: float = 2.0,
        max_wavelength_mm: float = 512.0,
    ) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands must be positive.")
        if not 0.0 < min_wavelength_mm <= max_wavelength_mm:
            raise ValueError("Invalid wavelength range.")

        wavelengths = torch.logspace(
            torch.log10(torch.tensor(min_wavelength_mm)),
            torch.log10(torch.tensor(max_wavelength_mm)),
            num_bands,
        )
        self.register_buffer("wavelengths_mm", wavelengths)
        self.projection = nn.Linear(3 * 2 * num_bands, embed_dim)

    def forward(
        self,
        grid_shape: Tuple[int, int, int],
        grid_spacing_mm: torch.Tensor,
    ) -> torch.Tensor:
        if grid_spacing_mm.ndim != 2 or grid_spacing_mm.shape[1] != 3:
            raise ValueError(
                "grid_spacing_mm must be [B, 3], got "
                f"{tuple(grid_spacing_mm.shape)}."
            )

        axes = [
            torch.arange(
                size,
                device=grid_spacing_mm.device,
                dtype=torch.float32,
            )
            - (size - 1) / 2.0
            for size in grid_shape
        ]
        grid = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=-1,
        ).reshape(1, -1, 3)
        coordinates_mm = grid * grid_spacing_mm[:, None, :]

        wavelengths = self.wavelengths_mm.to(grid_spacing_mm.device)
        phase = (
            2.0
            * torch.pi
            * coordinates_mm[..., None]
            / wavelengths[None, None, None, :]
        )
        fourier = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.projection(fourier.flatten(start_dim=2))


class SpacingAwarePatchEmbed3d(nn.Module):
    """Shared physical projection followed by a masked channel mean."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Int3 = (32, 32, 32),
        patch_kernel_size: Int3 = 8,
        init_sigma_mm: float = 8.0,
        learn_sigma: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = to_3tuple(grid_size)

        self.projection = PhysicalGaussianMaskedConv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_kernel_size,
            grid_size=self.grid_size,
            init_sigma_mm=init_sigma_mm,
            learn_sigma=learn_sigma,
        )
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _merge_channels(
        channel_tokens: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average valid channels independently at every spatial token."""

        batch_size, channels = channel_tokens.shape[:2]
        if channel_mask.shape != (batch_size, channels):
            raise ValueError(
                "channel_mask must have shape "
                f"{(batch_size, channels)}, got {tuple(channel_mask.shape)}."
            )
        if not channel_mask.any(dim=1).all():
            raise ValueError(
                "Every subject must contain at least one valid channel."
            )
        weights = channel_mask[:, :, None, None].to(channel_tokens.dtype)
        numerator = (channel_tokens * weights).sum(dim=1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_embedding: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        valid_spatial_shapes: Optional[
            Union[torch.Tensor, Sequence[Sequence[int]]]
        ] = None,
    ) -> PatchEmbedOutput:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, H, W, D], got {x.shape}.")
        batch_size, channels, height, width, depth = x.shape
        if channels < 1:
            raise ValueError("x must contain at least one input channel.")

        spacing = expand_spacing(spacing, batch_size, x.device)
        channel_images = x.reshape(
            batch_size * channels,
            1,
            height,
            width,
            depth,
        )
        channel_spacing = spacing.repeat_interleave(channels, dim=0)

        channel_shapes = None
        if valid_spatial_shapes is not None:
            subject_shapes = torch.as_tensor(
                valid_spatial_shapes,
                dtype=torch.long,
                device="cpu",
            )
            if subject_shapes.shape != (batch_size, 3):
                raise ValueError(
                    "valid_spatial_shapes must be [B, 3], got "
                    f"{tuple(subject_shapes.shape)}."
                )
            channel_shapes = subject_shapes.repeat_interleave(channels, dim=0)

        projection = self.projection(
            channel_images,
            channel_spacing,
            valid_spatial_shapes=channel_shapes,
            return_metadata=True,
        )
        channel_tokens = projection.features.flatten(2).transpose(1, 2)
        channel_tokens = channel_tokens.reshape(
            batch_size,
            channels,
            -1,
            channel_tokens.shape[-1],
        )

        if modality_embedding is not None:
            expected = (batch_size, channels, channel_tokens.shape[-1])
            if modality_embedding.shape != expected:
                raise ValueError(
                    f"modality_embedding must be {expected}, got "
                    f"{tuple(modality_embedding.shape)}."
                )
            channel_tokens = channel_tokens + modality_embedding.to(
                channel_tokens.dtype
            ).unsqueeze(2)

        if channel_mask is None:
            channel_mask = torch.ones(
                batch_size,
                channels,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            channel_mask = torch.as_tensor(
                channel_mask,
                device=x.device,
                dtype=torch.bool,
            )

        tokens = self._merge_channels(channel_tokens, channel_mask)
        grid_spacing_mm = projection.grid_spacing_mm.reshape(
            batch_size,
            channels,
            3,
        )[:, 0]
        return PatchEmbedOutput(
            tokens=self.norm(tokens),
            grid_shape=projection.grid_shape,
            grid_spacing_mm=grid_spacing_mm,
        )


class Mlp(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OffsetWindowAttention3d(nn.Module):
    """Non-cyclic 3D window attention with optional offset partitions."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Int3,
        offset: Int3,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.window_size = to_3tuple(window_size)
        self.offset = _nonnegative_3tuple(offset)
        if any(
            shift >= window
            for shift, window in zip(self.offset, self.window_size)
        ):
            raise ValueError("Every offset must be smaller than its window.")
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

    @staticmethod
    def _partition(
        x: torch.Tensor,
        window_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, height, width, depth, channels = x.shape
        wh, ww, wd = window_size
        return (
            x.reshape(
                batch_size,
                height // wh,
                wh,
                width // ww,
                ww,
                depth // wd,
                wd,
                channels,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(-1, wh * ww * wd, channels)
        )

    @staticmethod
    def _unpartition(
        windows: torch.Tensor,
        batch_size: int,
        padded_shape: Tuple[int, int, int],
        window_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        height, width, depth = padded_shape
        wh, ww, wd = window_size
        channels = windows.shape[-1]
        return (
            windows.reshape(
                batch_size,
                height // wh,
                width // ww,
                depth // wd,
                wh,
                ww,
                wd,
                channels,
            )
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .reshape(batch_size, height, width, depth, channels)
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        if num_tokens != grid_shape[0] * grid_shape[1] * grid_shape[2]:
            raise ValueError("Token count and grid shape are inconsistent.")

        height, width, depth = grid_shape
        wh, ww, wd = self.window_size
        oh, ow, od = self.offset
        after_h = (wh - (height + oh) % wh) % wh
        after_w = (ww - (width + ow) % ww) % ww
        after_d = (wd - (depth + od) % wd) % wd

        feature_map = x.reshape(batch_size, height, width, depth, channels)
        feature_map = feature_map.permute(0, 4, 1, 2, 3)
        feature_map = F.pad(
            feature_map,
            (od, after_d, ow, after_w, oh, after_h),
        ).permute(0, 2, 3, 4, 1)

        valid = torch.ones(
            batch_size,
            1,
            height,
            width,
            depth,
            device=x.device,
            dtype=x.dtype,
        )
        valid = F.pad(
            valid,
            (od, after_d, ow, after_w, oh, after_h),
        ).permute(0, 2, 3, 4, 1)

        padded_shape = (
            height + oh + after_h,
            width + ow + after_w,
            depth + od + after_d,
        )
        windows = self._partition(feature_map, self.window_size)
        valid_windows = self._partition(valid, self.window_size).squeeze(-1)
        key_padding_mask = ~valid_windows.bool()
        attended = self.attention(
            windows,
            windows,
            windows,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        attended = attended * valid_windows.unsqueeze(-1)
        attended_map = self._unpartition(
            attended,
            batch_size,
            padded_shape,
            self.window_size,
        )
        attended_map = attended_map[
            :,
            oh : oh + height,
            ow : ow + width,
            od : od + depth,
            :,
        ]
        return attended_map.reshape(batch_size, num_tokens, channels)


class WindowedTransformerBlock3d(nn.Module):
    """
    Local window attention plus bidirectional patch/global communication.

    Patch attention is O(N * window_volume), not O(N^2). Global tokens attend
    to a pooled spatial summary and every patch then cross-attends to the small
    global-token set. This keeps a true global pathway at a large 3D grid.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Int3 = (4, 4, 4),
        offset: Int3 = (0, 0, 0),
        summary_grid_size: Int3 = (8, 8, 8),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.summary_grid_size = to_3tuple(summary_grid_size)

        self.patch_norm1 = nn.LayerNorm(embed_dim)
        self.local_attention = OffsetWindowAttention3d(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            offset=offset,
            attention_dropout=attention_dropout,
        )

        self.global_query_norm = nn.LayerNorm(embed_dim)
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.global_from_patches = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.global_norm2 = nn.LayerNorm(embed_dim)
        self.global_mlp = Mlp(embed_dim, mlp_ratio, dropout)

        self.patch_query_norm = nn.LayerNorm(embed_dim)
        self.global_key_norm = nn.LayerNorm(embed_dim)
        self.patches_from_global = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.patch_norm2 = nn.LayerNorm(embed_dim)
        self.patch_mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def _summarize(
        self,
        patches: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, _, channels = patches.shape
        feature_map = patches.reshape(batch_size, *grid_shape, channels)
        feature_map = feature_map.permute(0, 4, 1, 2, 3)
        summary = F.adaptive_avg_pool3d(
            feature_map,
            self.summary_grid_size,
        )
        return summary.flatten(2).transpose(1, 2)

    def forward(
        self,
        patches: torch.Tensor,
        global_tokens: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = patches + self.local_attention(
            self.patch_norm1(patches),
            grid_shape,
        )

        summaries = self.summary_norm(self._summarize(patches, grid_shape))
        normalized_global = self.global_query_norm(global_tokens)
        global_tokens = global_tokens + self.global_from_patches(
            normalized_global,
            summaries,
            summaries,
            need_weights=False,
        )[0]
        global_tokens = global_tokens + self.global_mlp(
            self.global_norm2(global_tokens)
        )

        normalized_patches = self.patch_query_norm(patches)
        normalized_global = self.global_key_norm(global_tokens)
        patches = patches + self.patches_from_global(
            normalized_patches,
            normalized_global,
            normalized_global,
            need_weights=False,
        )[0]
        patches = patches + self.patch_mlp(self.patch_norm2(patches))
        return patches, global_tokens


class SpacingAwareViT3d(nn.Module):
    """Spacing-aware 3D ViT designed for a configurable fixed token lattice."""

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        grid_size: Int3 = (32, 32, 32),
        patch_kernel_size: Int3 = 8,
        window_size: Int3 = (4, 4, 4),
        summary_grid_size: Int3 = (8, 8, 8),
        num_register_tokens: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        init_sigma_mm: float = 8.0,
        learn_sigma: bool = True,
        position_num_bands: int = 16,
        intermediate_layers: Optional[Sequence[int]] = None,
        modality_to_id: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens cannot be negative.")

        self.embed_dim = int(embed_dim)
        self.grid_size = to_3tuple(grid_size)
        self.num_register_tokens = int(num_register_tokens)
        window_size = to_3tuple(window_size)
        summary_grid_size = to_3tuple(summary_grid_size)

        self.patch_embed = SpacingAwarePatchEmbed3d(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            patch_kernel_size=patch_kernel_size,
            init_sigma_mm=init_sigma_mm,
            learn_sigma=learn_sigma,
        )
        self.position_embed = PhysicalFourierPositionEmbedding3d(
            embed_dim=embed_dim,
            num_bands=position_num_bands,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_tokens = nn.Parameter(
            torch.zeros(1, 1 + num_register_tokens, embed_dim)
        )
        self.global_position = nn.Parameter(
            torch.zeros(1, 1 + num_register_tokens, embed_dim)
        )

        modality_ids = dict(
            MODALITY_TO_ID if modality_to_id is None else modality_to_id
        )
        if modality_ids.get("UNKNOWN") != 0:
            raise ValueError("UNKNOWN modality must exist and use ID 0.")
        self.unknown_modality_id = 0
        self.num_modalities = max(modality_ids.values()) + 1
        self.padding_modality_id = self.num_modalities
        self.modality_embed = nn.Embedding(
            self.num_modalities + 1,
            embed_dim,
            padding_idx=self.unknown_modality_id,
        )
        with torch.no_grad():
            self.modality_embed.weight[self.unknown_modality_id].zero_()

        self.position_dropout = nn.Dropout(dropout)
        half_window = tuple(value // 2 for value in window_size)
        self.blocks = nn.ModuleList(
            WindowedTransformerBlock3d(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                offset=(0, 0, 0) if index % 2 == 0 else half_window,
                summary_grid_size=summary_grid_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for index in range(depth)
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.global_norm = nn.LayerNorm(embed_dim)

        if intermediate_layers is None:
            intermediate_layers = tuple(
                round(value)
                for value in torch.linspace(
                    0,
                    depth - 1,
                    steps=min(4, depth),
                ).tolist()
            )
        layers = tuple(sorted(set(int(value) for value in intermediate_layers)))
        if any(value < 0 or value >= depth for value in layers):
            raise ValueError(
                f"intermediate_layers must be in [0, {depth - 1}], got {layers}."
            )
        self.intermediate_layers = layers
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.global_tokens, std=0.02)
        nn.init.trunc_normal_(self.global_position, std=0.02)

    @staticmethod
    def tokens_to_map(
        tokens: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = tokens.shape
        if num_tokens != grid_shape[0] * grid_shape[1] * grid_shape[2]:
            raise ValueError("Token count and grid shape are inconsistent.")
        return (
            tokens.reshape(batch_size, *grid_shape, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def _prepare_channel_metadata(
        self,
        x: torch.Tensor,
        modality: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels = x.shape[:2]
        if channel_mask is None:
            channel_mask = torch.ones(
                batch_size,
                channels,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            channel_mask = torch.as_tensor(
                channel_mask,
                device=x.device,
                dtype=torch.bool,
            )
        if channel_mask.shape != (batch_size, channels):
            raise ValueError(
                "channel_mask must be "
                f"{(batch_size, channels)}, got {tuple(channel_mask.shape)}."
            )
        if not channel_mask.any(dim=1).all():
            raise ValueError("Every subject requires at least one real channel.")

        if modality is None:
            modality = torch.full(
                (batch_size, channels),
                self.unknown_modality_id,
                device=x.device,
                dtype=torch.long,
            )
        else:
            modality = torch.as_tensor(
                modality,
                device=x.device,
                dtype=torch.long,
            )
        if modality.shape != (batch_size, channels):
            raise ValueError(
                f"modality must be {(batch_size, channels)}, got "
                f"{tuple(modality.shape)}."
            )

        safe_modality = modality.masked_fill(
            ~channel_mask,
            self.padding_modality_id,
        )
        real_modality = safe_modality[channel_mask]
        if (
            (real_modality < 0).any()
            or (real_modality >= self.num_modalities).any()
        ):
            raise ValueError("A real channel contains an invalid modality ID.")
        return safe_modality, channel_mask

    def forward_features(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid_spatial_shapes: Optional[
            Union[torch.Tensor, Sequence[Sequence[int]]]
        ] = None,
        return_feature_map: bool = False,
        return_intermediate: bool = False,
    ) -> ViTFeatures:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, H, W, D], got {x.shape}.")
        batch_size = x.shape[0]
        spacing = expand_spacing(spacing, batch_size, x.device)
        safe_modality, channel_mask = self._prepare_channel_metadata(
            x,
            modality,
            channel_mask,
        )
        modality_embedding = self.modality_embed(safe_modality)
        modality_embedding = modality_embedding * channel_mask.unsqueeze(-1).to(
            modality_embedding.dtype
        )

        embedded = self.patch_embed(
            x,
            spacing,
            modality_embedding=modality_embedding,
            channel_mask=channel_mask,
            valid_spatial_shapes=valid_spatial_shapes,
        )
        patches = embedded.tokens
        if mask is not None:
            if mask.shape != patches.shape[:2]:
                raise ValueError(
                    f"mask must be {patches.shape[:2]}, got {tuple(mask.shape)}."
                )
            patches = torch.where(
                mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1),
                self.mask_token.to(dtype=patches.dtype),
                patches,
            )

        position = self.position_embed(
            embedded.grid_shape,
            embedded.grid_spacing_mm,
        ).to(patches.dtype)
        patches = self.position_dropout(patches + position)
        global_tokens = (
            self.global_tokens + self.global_position
        ).expand(batch_size, -1, -1)
        global_tokens = self.position_dropout(global_tokens)

        intermediate: List[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            patches, global_tokens = block(
                patches,
                global_tokens,
                embedded.grid_shape,
            )
            if return_intermediate and index in self.intermediate_layers:
                intermediate.append(
                    self.tokens_to_map(
                        self.patch_norm(patches),
                        embedded.grid_shape,
                    )
                )

        patches = self.patch_norm(patches)
        global_tokens = self.global_norm(global_tokens)
        feature_map = (
            self.tokens_to_map(patches, embedded.grid_shape)
            if return_feature_map
            else None
        )
        return ViTFeatures(
            cls=global_tokens[:, 0],
            registers=global_tokens[:, 1:],
            patches=patches,
            patch_grid_shape=embedded.grid_shape,
            grid_spacing_mm=embedded.grid_spacing_mm,
            feature_map=feature_map,
            intermediate_feature_maps=tuple(intermediate),
        )

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid_spatial_shapes: Optional[
            Union[torch.Tensor, Sequence[Sequence[int]]]
        ] = None,
        return_feature_map: bool = False,
        return_intermediate: bool = False,
    ) -> Dict[str, object]:
        return self.forward_features(
            x=x,
            spacing=spacing,
            modality=modality,
            channel_mask=channel_mask,
            mask=mask,
            valid_spatial_shapes=valid_spatial_shapes,
            return_feature_map=return_feature_map,
            return_intermediate=return_intermediate,
        ).as_dict()


class ProjectionHead(nn.Module):
    """DINO-style projection head with a weight-normalized output layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")

        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: List[nn.Module] = [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ]
            for _ in range(num_layers - 2):
                layers.extend(
                    (nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                )
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        return self.last_layer(x)


class BrainDinoViT(nn.Module):
    """
    BrainDINO student/teacher model with memory-aware iBOT projection.

    At a large 3D grid, dense patch logits are very large. ``patch_projection_mask``
    therefore selects only required iBOT positions and returns flat projections
    plus [batch_index, token_index] coordinates. Use the same selection mask for
    the masked student and unmasked teacher.
    """

    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        cls_projection_dim: int = 8192,
        patch_projection_dim: int = 1024,
        projection_hidden_dim: int = 2048,
        projection_bottleneck_dim: int = 256,
        projection_num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        head_kwargs = {
            "hidden_dim": projection_hidden_dim,
            "bottleneck_dim": projection_bottleneck_dim,
            "num_layers": projection_num_layers,
        }
        self.cls_head = ProjectionHead(
            backbone.embed_dim,
            cls_projection_dim,
            **head_kwargs,
        )
        self.patch_head = ProjectionHead(
            backbone.embed_dim,
            patch_projection_dim,
            **head_kwargs,
        )

    def _project_selected_patches(
        self,
        patches: torch.Tensor,
        selection_mask: Optional[torch.Tensor],
        project_all_patches: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if selection_mask is None:
            if not project_all_patches:
                return None, None
            return self.patch_head(patches), None

        if selection_mask.shape != patches.shape[:2]:
            raise ValueError(
                "patch_projection_mask must be "
                f"{patches.shape[:2]}, got {tuple(selection_mask.shape)}."
            )
        indices = selection_mask.to(
            device=patches.device,
            dtype=torch.bool,
        ).nonzero(as_tuple=False)
        if indices.numel() == 0:
            empty = patches.new_empty((0, self.patch_head.out_dim))
            return empty, indices
        selected = patches[indices[:, 0], indices[:, 1]]
        return self.patch_head(selected), indices

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        patch_projection_mask: Optional[torch.Tensor] = None,
        valid_spatial_shapes: Optional[
            Union[torch.Tensor, Sequence[Sequence[int]]]
        ] = None,
        mode: PretrainingMode = "student",
        project_all_patches: bool = False,
        return_backbone_features: bool = True,
    ) -> Dict[str, object]:
        if mode not in ("student", "teacher", "features"):
            raise ValueError(f"Unsupported mode: {mode}.")
        if mode == "teacher" and mask is not None:
            raise ValueError("Teacher inputs must remain unmasked.")

        backbone_mask = mask if mode == "student" else None
        features = self.backbone.forward_features(
            x=x,
            spacing=spacing,
            modality=modality,
            channel_mask=channel_mask,
            mask=backbone_mask,
            valid_spatial_shapes=valid_spatial_shapes,
            return_feature_map=False,
            return_intermediate=False,
        )
        if mode == "features":
            return features.as_dict()

        if patch_projection_mask is None and mode == "student":
            patch_projection_mask = mask
        patch_projection, patch_indices = self._project_selected_patches(
            features.patches,
            patch_projection_mask,
            project_all_patches,
        )
        output: Dict[str, object] = {
            "cls_projection": self.cls_head(features.cls),
            "patch_projection": patch_projection,
            "patch_projection_indices": patch_indices,
            "patch_grid_shape": features.patch_grid_shape,
            "grid_spacing_mm": features.grid_spacing_mm,
        }
        if return_backbone_features:
            output.update(
                {
                    "cls_features": features.cls,
                    "register_features": features.registers,
                    "patch_features": features.patches,
                }
            )
        return output


def pretrain_braindino_vit_s(
    grid_size: Int3 = (32, 32, 32),
    patch_kernel_size: Int3 = 8,
    window_size: Int3 = (4, 4, 4),
    summary_grid_size: Int3 = (8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    cls_projection_dim: int = 8192,
    patch_projection_dim: int = 1024,
) -> BrainDinoViT:
    backbone = SpacingAwareViT3d(
        embed_dim=384,
        depth=12,
        num_heads=6,
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        num_register_tokens=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
        position_num_bands=16,
        intermediate_layers=(2, 5, 8, 11),
    )
    return BrainDinoViT(
        backbone=backbone,
        cls_projection_dim=cls_projection_dim,
        patch_projection_dim=patch_projection_dim,
        projection_hidden_dim=2048,
        projection_bottleneck_dim=256,
        projection_num_layers=3,
    )


def pretrain_braindino_vit_b(
    grid_size: Int3 = (32, 32, 32),
    patch_kernel_size: Int3 = 8,
    window_size: Int3 = (4, 4, 4),
    summary_grid_size: Int3 = (8, 8, 8),
    init_sigma_mm: float = 8.0,
    learn_sigma: bool = True,
    cls_projection_dim: int = 16384,
    patch_projection_dim: int = 2048,
) -> BrainDinoViT:
    backbone = SpacingAwareViT3d(
        embed_dim=768,
        depth=12,
        num_heads=12,
        grid_size=grid_size,
        patch_kernel_size=patch_kernel_size,
        window_size=window_size,
        summary_grid_size=summary_grid_size,
        num_register_tokens=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,
        position_num_bands=16,
        intermediate_layers=(2, 5, 8, 11),
    )
    return BrainDinoViT(
        backbone=backbone,
        cls_projection_dim=cls_projection_dim,
        patch_projection_dim=patch_projection_dim,
        projection_hidden_dim=3072,
        projection_bottleneck_dim=384,
        projection_num_layers=3,
    )


def _smoke_test() -> None:
    torch.manual_seed(0)
    backbone = SpacingAwareViT3d(
        embed_dim=48,
        depth=2,
        num_heads=4,
        grid_size=(8, 8, 8),
        patch_kernel_size=3,
        window_size=(4, 4, 4),
        summary_grid_size=(2, 2, 2),
        num_register_tokens=2,
        intermediate_layers=(0, 1),
        modality_to_id={"UNKNOWN": 0, "T1": 1, "FLAIR": 2},
    )
    model = BrainDinoViT(
        backbone,
        cls_projection_dim=64,
        patch_projection_dim=32,
        projection_hidden_dim=96,
        projection_bottleneck_dim=24,
    ).eval()
    x = torch.randn(2, 2, 32, 32, 8)
    spacing = torch.tensor(((1.0, 1.0, 4.0), (0.8, 0.8, 5.0)))
    modality = torch.tensor(((1, 2), (1, 2)))
    mask = torch.zeros(2, 8 * 8 * 8, dtype=torch.bool)
    mask[:, ::4] = True

    with torch.inference_mode():
        student = model(
            x,
            spacing,
            modality=modality,
            mask=mask,
            mode="student",
        )
        teacher = model(
            x,
            spacing,
            modality=modality,
            patch_projection_mask=mask,
            mode="teacher",
        )
    assert student["cls_projection"].shape == (2, 64)
    assert student["patch_projection"].shape == (256, 32)
    assert teacher["patch_projection"].shape == (256, 32)
    assert torch.equal(
        student["patch_projection_indices"],
        teacher["patch_projection_indices"],
    )


if __name__ == "__main__":
    _smoke_test()
    print("BrainDinoViT smoke test passed")
