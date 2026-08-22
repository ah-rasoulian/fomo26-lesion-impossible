from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .blocks.layers.physical_conv3d import PhysicalGaussianMaskedConv3d
from .blocks.utils import MODALITY_TO_ID


Int3 = Union[int, Sequence[int]]


def _triple(value: Int3) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    if len(value) != 3:
        raise ValueError(f"Expected three values, got {value}.")
    return tuple(int(v) for v in value)


def _expand_spacing(
    spacing: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    spacing = torch.as_tensor(spacing, device=device, dtype=torch.float32)
    if spacing.shape == (3,):
        spacing = spacing.unsqueeze(0).expand(batch_size, -1)
    if spacing.shape != (batch_size, 3):
        raise ValueError(
            f"spacing must have shape [3] or [B, 3], got {tuple(spacing.shape)}."
        )
    return spacing


@dataclass
class ViTFeatures:
    """Decoder-facing representation returned by the backbone."""

    cls: torch.Tensor
    patches: torch.Tensor
    patch_grid_shape: Tuple[int, int, int]
    feature_map: torch.Tensor
    intermediate_feature_maps: Tuple[torch.Tensor, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "cls_features": self.cls,
            "patch_features": self.patches,
            "patch_grid_shape": self.patch_grid_shape,
            "feature_map": self.feature_map,
            "intermediate_feature_maps": self.intermediate_feature_maps,
        }


class PhysicalFourierPositionEmbedding3d(nn.Module):
    """
    Continuous positional encoding evaluated at patch centers in millimetres.

    Unlike a learned position table, this supports arbitrary crop and token-grid
    sizes. Spacing affects both the patch convolution and the token positions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_bands: int = 16,
        min_wavelength_mm: float = 2.0,
        max_wavelength_mm: float = 256.0,
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
        spacing: torch.Tensor,
        patch_stride: Tuple[int, int, int],
    ) -> torch.Tensor:
        axes = [
            torch.arange(size, device=spacing.device, dtype=torch.float32)
            * stride
            for size, stride in zip(grid_shape, patch_stride)
        ]
        grid = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=-1,
        ).reshape(1, -1, 3)

        # Centering avoids encoding an arbitrary image origin while retaining
        # the physical extent and relative displacement of the patch centers.
        grid = grid - grid.mean(dim=1, keepdim=True)
        coordinates_mm = grid * spacing[:, None, :]
        phase = (
            2.0
            * torch.pi
            * coordinates_mm[..., None]
            / self.wavelengths_mm.to(spacing.device)[None, None, None, :]
        )
        fourier = torch.cat((phase.sin(), phase.cos()), dim=-1)
        fourier = fourier.flatten(start_dim=2)
        return self.projection(fourier)


class SpacingAwarePatchEmbed3d(nn.Module):
    """
    Channel-count-invariant spacing-aware patch embedding.

    A shared single-channel physical convolution processes every channel.
    Each channel receives its own modality embedding before channel features
    are mean-pooled. Parameters therefore do not depend on the input channel
    count.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_kernel_size: Int3 = 7,
        patch_stride: Int3 = 4,
        patch_padding: Optional[Int3] = None,
        init_sigma_mm: float = 3.0,
        learn_sigma: bool = True,
        channel_fusion_num_heads: int = 6,
        channel_fusion_mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim % channel_fusion_num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by channel_fusion_num_heads."
            )

        kernel = _triple(patch_kernel_size)
        if any(value % 2 == 0 for value in kernel):
            raise ValueError(
                "PhysicalGaussianMaskedConv3d requires odd patch kernel sizes."
            )
        self.patch_stride = _triple(patch_stride)
        self.projection = PhysicalGaussianMaskedConv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=kernel,
            stride=self.patch_stride,
            padding=patch_padding,
            init_sigma_mm=init_sigma_mm,
            learn_sigma=learn_sigma,
        )
        self.channel_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        channel_hidden_dim = int(
            embed_dim * channel_fusion_mlp_ratio
        )

        self.channel_attention_norm = nn.LayerNorm(embed_dim)
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=channel_fusion_num_heads,
            batch_first=True,
        )

        self.channel_attention_mlp_norm = nn.LayerNorm(embed_dim)
        self.channel_attention_mlp = nn.Sequential(
            nn.Linear(embed_dim, channel_hidden_dim),
            nn.GELU(),
            nn.Linear(channel_hidden_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def _fuse_channels(
            self,
            channel_tokens: torch.Tensor,
            channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            channel_tokens: [B, C, N, E]
            channel_mask:   [B, C]

        Returns:
            Subject-level tokens: [B, N, E]
        """
        batch_size, channels, num_patches, embed_dim = (
            channel_tokens.shape
        )

        if channel_mask.shape != (batch_size, channels):
            raise ValueError(
                "channel_mask must have shape "
                f"{(batch_size, channels)}, "
                f"got {tuple(channel_mask.shape)}."
            )

        if not channel_mask.any(dim=1).all():
            raise ValueError(
                "Every sample must contain at least one valid channel."
            )

        # Attend across channels independently at every spatial patch.
        fusion_tokens = (
            channel_tokens
            .permute(0, 2, 1, 3)
            .reshape(
                batch_size * num_patches,
                channels,
                embed_dim,
            )
        )

        padding_mask = (
            ~channel_mask[:, None, :]
            .expand(batch_size, num_patches, channels)
            .reshape(batch_size * num_patches, channels)
        )

        normalized = self.channel_attention_norm(fusion_tokens)

        attention_chunk_size = 4096
        attended_chunks = []

        for start in range(0, normalized.shape[0], attention_chunk_size):
            end = min(start + attention_chunk_size, normalized.shape[0])

            chunk = normalized[start:end]

            chunk_padding_mask = padding_mask[start:end] if padding_mask is not None else None

            chunk_attended = self.channel_attention(
                chunk,
                chunk,
                chunk,
                key_padding_mask=chunk_padding_mask,
                need_weights=False,
            )[0]

            attended_chunks.append(chunk_attended)

        attended = torch.cat(attended_chunks, dim=0)

        fusion_tokens = fusion_tokens + attended
        fusion_tokens = fusion_tokens + self.channel_attention_mlp(
            self.channel_attention_mlp_norm(fusion_tokens)
        )

        fusion_tokens = fusion_tokens.reshape(
            batch_size,
            num_patches,
            channels,
            embed_dim,
        )

        weights = channel_mask[
            :, None, :, None
        ].to(fusion_tokens.dtype)

        return (
                fusion_tokens * weights
        ).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_embedding: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, H, W, D], got {x.shape}.")
        batch_size, channels, height, width, depth = x.shape
        if channels < 1:
            raise ValueError("x must contain at least one input channel.")

        spacing = _expand_spacing(spacing, batch_size, x.device)
        channel_images = x.reshape(
            batch_size * channels, 1, height, width, depth
        )
        channel_spacing = spacing.repeat_interleave(channels, dim=0)
        channel_features = self.projection(channel_images, channel_spacing)

        grid_shape = tuple(int(v) for v in channel_features.shape[2:])
        channel_tokens = channel_features.flatten(2).transpose(1, 2)
        channel_tokens = channel_tokens.reshape(
            batch_size,
            channels,
            -1,
            channel_tokens.shape[-1],
        )

        if modality_embedding is not None:
            expected_shape = (batch_size, channels, channel_tokens.shape[-1])
            if modality_embedding.shape != expected_shape:
                raise ValueError(
                    "modality_embedding must be [B, C, E]="
                    f"{expected_shape}, got {modality_embedding.shape}."
                )
            channel_tokens = (
                channel_tokens
                + modality_embedding.to(dtype=channel_tokens.dtype).unsqueeze(2)
            )

        # Nonlinear shared processing binds each channel's image features to
        # its modality identity before the set aggregation. Without this step,
        # mean(image + modality) would lose the image-modality correspondence.
        channel_tokens = channel_tokens + self.channel_fusion(channel_tokens)

        if channel_mask is None:
            channel_mask = torch.ones(
                (batch_size, channels),
                device=channel_tokens.device,
                dtype=torch.bool,
            )
        else:
            channel_mask = torch.as_tensor(
                channel_mask,
                device=channel_tokens.device,
                dtype=torch.bool,
            )

        tokens = self._fuse_channels(
            channel_tokens=channel_tokens,
            channel_mask=channel_mask,
        )

        return self.norm(tokens), grid_shape


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        return x + self.mlp(self.norm2(x))


class SpacingAwareViT3d(nn.Module):
    """
    Decoder-agnostic 3D ViT backbone.

    Input convention is [B, C, H, W, D], matching
    PhysicalGaussianMaskedConv3d. The names of the three spatial axes do not
    affect PyTorch operations, but spacing must use the same axis order.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_kernel_size: Int3 = 7,
        patch_stride: Int3 = 4,
        patch_padding: Optional[Int3] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        init_sigma_mm: float = 3.0,
        learn_sigma: bool = True,
        position_num_bands: int = 16,
        intermediate_layers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.patch_embed = SpacingAwarePatchEmbed3d(
            embed_dim=embed_dim,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            init_sigma_mm=init_sigma_mm,
            learn_sigma=learn_sigma,
            channel_fusion_num_heads=num_heads,
        )
        self.position_embed = PhysicalFourierPositionEmbedding3d(
            embed_dim=embed_dim,
            num_bands=position_num_bands,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_position = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.unknown_modality_id = MODALITY_TO_ID["UNKNOWN"]

        if self.unknown_modality_id != 0:
            raise ValueError("UNKNOWN modality must use ID 0.")

        # Robust even if modality IDs are not perfectly contiguous.
        self.num_modalities = max(MODALITY_TO_ID.values()) + 1
        self.padding_modality_id = self.num_modalities

        self.modality_embed = nn.Embedding(
            num_embeddings=self.num_modalities + 1,
            embedding_dim=embed_dim,
            padding_idx=self.unknown_modality_id,
        )

        with torch.no_grad():
            # UNKNOWN contributes no modality-specific information.
            self.modality_embed.weight[self.unknown_modality_id].zero_()

        self.position_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        if intermediate_layers is None:
            intermediate_layers = tuple(
                round(value)
                for value in torch.linspace(0, depth - 1, steps=min(4, depth)).tolist()
            )
        layers = tuple(sorted(set(int(index) for index in intermediate_layers)))
        if any(index < 0 or index >= depth for index in layers):
            raise ValueError(
                f"intermediate_layers must be in [0, {depth - 1}], got {layers}."
            )
        self.intermediate_layers = layers
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls_position, std=0.02)

    @staticmethod
    def tokens_to_map(
        tokens: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = tokens.shape
        if num_tokens != grid_shape[0] * grid_shape[1] * grid_shape[2]:
            raise ValueError("Token count and patch grid shape are inconsistent.")
        return (
            tokens.reshape(batch_size, *grid_shape, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def forward_features(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> ViTFeatures:
        if x.ndim != 5:
            raise ValueError(f"x must be 5D [B, C, H, W, D], got {x.shape}.")
        batch_size, channels = x.shape[:2]
        spacing = _expand_spacing(spacing, batch_size, x.device)

        if channel_mask is None:
            channel_mask = torch.ones(
                (batch_size, channels),
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
                "channel_mask must have shape "
                f"{(batch_size, channels)}, got {tuple(channel_mask.shape)}."
            )

        if not channel_mask.any(dim=1).all():
            raise ValueError(
                "Every sample must contain at least one real image channel."
            )

        if modality is None:
            modality = torch.full(
                (batch_size, channels),
                fill_value=self.unknown_modality_id,
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
                "modality must have shape "
                f"{(batch_size, channels)}, got {tuple(modality.shape)}."
            )

        # Convert the collator's -1 values into a valid embedding index.
        safe_modality = modality.masked_fill(
            ~channel_mask,
            self.padding_modality_id,
        )

        # Reject invalid IDs on real channels.
        real_modality = safe_modality[channel_mask]
        if (
                (real_modality < 0).any()
                or (real_modality >= self.num_modalities).any()
        ):
            raise ValueError("A real channel contains an invalid modality ID.")

        modality_embedding = self.modality_embed(safe_modality)
        modality_embedding = (
                modality_embedding
                * channel_mask.unsqueeze(-1).to(modality_embedding.dtype)
        )

        patches, grid_shape = self.patch_embed(
            x,
            spacing,
            modality_embedding=modality_embedding,
            channel_mask=channel_mask,
        )

        if mask is not None:
            if mask.shape != patches.shape[:2]:
                raise ValueError(
                    f"mask must be [B, N]={patches.shape[:2]}, got {mask.shape}."
                )
            patches = torch.where(
                mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1),
                self.mask_token.to(dtype=patches.dtype),
                patches,
            )

        patch_position = self.position_embed(
            grid_shape,
            spacing,
            self.patch_embed.patch_stride,
        ).to(dtype=patches.dtype)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat(
            (cls + self.cls_position, patches + patch_position),
            dim=1,
        )
        tokens = self.position_dropout(tokens)

        intermediate: List[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.intermediate_layers:
                intermediate.append(
                    self.tokens_to_map(
                        self.norm(tokens[:, 1:]),
                        grid_shape,
                    )
                )

        tokens = self.norm(tokens)
        patch_tokens = tokens[:, 1:]
        return ViTFeatures(
            cls=tokens[:, 0],
            patches=patch_tokens,
            patch_grid_shape=grid_shape,
            feature_map=self.tokens_to_map(patch_tokens, grid_shape),
            intermediate_feature_maps=tuple(intermediate),
        )

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        return self.forward_features(x, spacing, modality, channel_mask, mask).as_dict()


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.normalize(self.mlp(x), dim=-1)
        return self.last_layer(x)


class BrainDinoViT(nn.Module):
    """Adapter matching the exact output contract of BrainDinoModule."""

    def __init__(
        self,
        backbone: SpacingAwareViT3d,
        cls_projection_dim: int = 65536,
        patch_projection_dim: int = 8192,
        projection_hidden_dim: int = 2048,
        projection_bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        head_kwargs = {
            "hidden_dim": projection_hidden_dim,
            "bottleneck_dim": projection_bottleneck_dim,
        }
        self.cls_head = ProjectionHead(
            backbone.embed_dim, cls_projection_dim, **head_kwargs
        )
        self.patch_head = ProjectionHead(
            backbone.embed_dim, patch_projection_dim, **head_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        features = self.backbone.forward_features(x, spacing, modality, channel_mask, mask)
        return {
            "cls_features": features.cls,
            "patch_features": features.patches,
            "cls_projection": self.cls_head(features.cls),
            "patch_projection": self.patch_head(features.patches),
            "patch_grid_shape": features.patch_grid_shape,
        }


def pretrain_braindino_vit_s(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,

    cls_projection_dim: int,
    patch_projection_dim: int,
):
    backbone = SpacingAwareViT3d(
        # ViT-S architecture
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,

        # Spacing-aware patch embedding
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        init_sigma_mm=init_sigma_mm,
        learn_sigma=learn_sigma,

        # Physical Fourier position encoding
        position_num_bands=16,

        # Regularization
        dropout=0.0,
        attention_dropout=0.0,

        # Useful later for segmentation decoders
        intermediate_layers=(2, 5, 8, 11),
    )

    return BrainDinoViT(
        backbone=backbone,

        # DINO and iBOT output spaces
        cls_projection_dim=cls_projection_dim,
        patch_projection_dim=patch_projection_dim,

        # Projection-head architecture
        projection_hidden_dim=2048,
        projection_bottleneck_dim=256,
    )

def pretrain_braindino_vit_b(
    patch_kernel_size: int,
    patch_stride: int,
    patch_padding: int,
    init_sigma_mm: float,
    learn_sigma: bool,
    cls_projection_dim: int,
    patch_projection_dim: int,
):
    backbone = SpacingAwareViT3d(
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

        dropout=0.0,
        attention_dropout=0.0,

        intermediate_layers=(2, 5, 8, 11),
    )

    return BrainDinoViT(
        backbone=backbone,

        cls_projection_dim=cls_projection_dim,
        patch_projection_dim=patch_projection_dim,

        projection_hidden_dim=2048,
        projection_bottleneck_dim=256,
    )


def _print_output_shapes(
    title: str,
    outputs: Dict[str, object],
) -> None:
    """Print tensor shapes and other metadata returned by a model."""
    print(f"\n{title}")
    for name, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {tuple(value.shape)}")
        elif isinstance(value, (tuple, list)) and all(
            isinstance(item, torch.Tensor) for item in value
        ):
            print(
                f"  {name}: "
                f"{[tuple(item.shape) for item in value]}"
            )
        else:
            print(f"  {name}: {value}")


def main() -> None:
    """Run a small end-to-end smoke test with a synthetic 3D batch."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    spatial_shape = (128, 128, 128)

    # The image and spacing axes use the same [H, W, D] order.
    x = torch.randn(batch_size, 2, *spatial_shape, device=device)
    x_single_channel = torch.randn(
        batch_size, 1, *spatial_shape, device=device
    )
    x_five_channels = torch.randn(
        batch_size, 5, *spatial_shape, device=device
    )
    spacing = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.8, 0.8, 2.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    # Each channel has its own modality identity: modality.shape == [B, C].
    modality = torch.tensor(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=torch.long,
        device=device,
    )
    single_channel_modality = torch.tensor(
        [
            [0],
            [3],
        ],
        dtype=torch.long,
        device=device,
    )
    five_channel_modality = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    # A deliberately small configuration keeps this test inexpensive on CPU.
    backbone = SpacingAwareViT3d(
        embed_dim=96,
        depth=4,
        num_heads=4,
        patch_kernel_size=7,
        patch_stride=4,
        intermediate_layers=(0, 1, 2, 3),
    ).to(device)
    backbone.eval()

    print(f"Device: {device}")
    print(f"Input image: {tuple(x.shape)}")
    print(f"Spacing: {spacing.cpu().tolist()}")
    print(f"Per-channel modality IDs: {modality.cpu().tolist()}")
    print(
        "Learned physical sigma (mm): "
        f"{backbone.patch_embed.projection.sigma_mm().item():.4f}"
    )

    with torch.inference_mode():
        backbone_outputs = backbone(x, spacing, modality)
        single_channel_outputs = backbone(
            x_single_channel, spacing, single_channel_modality
        )
        five_channel_outputs = backbone(
            x_five_channels, spacing, five_channel_modality
        )
        permuted_outputs = backbone(
            x.flip(1),
            spacing,
            modality.flip(1),
        )
    _print_output_shapes("Backbone outputs", backbone_outputs)
    print("\nArbitrary-channel checks using the same backbone")
    print(
        "  C=1 patch features: "
        f"{tuple(single_channel_outputs['patch_features'].shape)}"
    )
    print(
        "  C=2 patch features: "
        f"{tuple(backbone_outputs['patch_features'].shape)}"
    )
    print(
        "  C=5 patch features: "
        f"{tuple(five_channel_outputs['patch_features'].shape)}"
    )
    permutation_error = (
        backbone_outputs["patch_features"]
        - permuted_outputs["patch_features"]
    ).abs().max()
    print(
        "  joint channel/modality permutation max error: "
        f"{permutation_error.item():.3e}"
    )

    # Mask every fourth patch to exercise the masked-token/iBOT path.
    num_patches = backbone_outputs["patch_features"].shape[1]
    patch_mask = torch.zeros(
        batch_size,
        num_patches,
        dtype=torch.bool,
        device=device,
    )
    patch_mask[:, ::4] = True

    pretraining_model = BrainDinoViT(
        backbone=backbone,
        cls_projection_dim=128,
        patch_projection_dim=64,
        projection_hidden_dim=192,
        projection_bottleneck_dim=48,
    ).to(device)
    pretraining_model.eval()

    with torch.inference_mode():
        pretraining_outputs = pretraining_model(
            x,
            spacing,
            modality,
            mask=patch_mask,
        )
    _print_output_shapes(
        "DINO/iBOT pretraining outputs",
        pretraining_outputs,
    )
    print(f"  masked patches per sample: {patch_mask.sum(dim=1).cpu().tolist()}")


if __name__ == "__main__":
    main()