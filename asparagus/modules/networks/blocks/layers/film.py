from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import expand_spacing

class ChannelModalityEmbeddingStem(nn.Module):
    """
    Modality-aware input stem.

    Each image channel is projected independently using shared weights.
    A modality embedding modulates that channel.
    Then channels are aggregated into one feature tensor.

    Output:
        y: [B, output_channels, D, H, W]
        modality_context: [B, modality_embed_dim]
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_modalities: int,
        modality_embed_dim: int = 32,
        aggregation: str = "sqrt_sum",
        padding_idx: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_modalities = num_modalities
        self.modality_embed_dim = modality_embed_dim
        self.aggregation = aggregation

        self.value_proj = nn.Conv3d(
            in_channels=1,
            out_channels=output_channels,
            kernel_size=1,
            bias=bias,
        )

        self.modality_embedding = nn.Embedding(
            num_embeddings=num_modalities,
            embedding_dim=modality_embed_dim,
            padding_idx=padding_idx,
        )

        self.modality_to_gamma_beta = nn.Linear(
            modality_embed_dim,
            2 * output_channels,
        )

        nn.init.zeros_(self.modality_to_gamma_beta.weight)
        nn.init.zeros_(self.modality_to_gamma_beta.bias)

    def _prepare_modality_ids(
        self,
        modality_ids: torch.Tensor,
        B: int,
        C: int,
        device: torch.device,
    ) -> torch.Tensor:
        modality_ids = torch.as_tensor(modality_ids, dtype=torch.long, device=device)

        if modality_ids.ndim == 1:
            if modality_ids.shape[0] == C:
                modality_ids = modality_ids[None, :].expand(B, C)
            elif C == 1 and modality_ids.shape[0] == B:
                modality_ids = modality_ids[:, None]
            else:
                raise ValueError(
                    f"Ambiguous modality_ids shape {modality_ids.shape}. "
                    f"Expected [C], [B] when C=1, or [B, C]."
                )

        elif modality_ids.ndim == 2:
            if modality_ids.shape != (B, C):
                raise ValueError(
                    f"modality_ids must be [B, C]. Got {modality_ids.shape}, "
                    f"expected {(B, C)}"
                )
        else:
            raise ValueError(f"modality_ids must be [C], [B], or [B, C].")

        return modality_ids

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, H, W, D], got {x.shape}")

        B, C, H, W, D = x.shape

        if C > self.input_channels:
            raise ValueError(
                f"Input has {C} channels, but encoder was initialized "
                f"with input_channels={self.input_channels}"
            )

        modality_ids = self._prepare_modality_ids(
            modality_ids=modality_ids,
            B=B,
            C=C,
            device=x.device,
        )

        if modality_mask is None:
            modality_mask = torch.ones(B, C, device=x.device, dtype=torch.float32)
        else:
            modality_mask = modality_mask.to(device=x.device, dtype=torch.float32)
            if modality_mask.shape != (B, C):
                raise ValueError(
                    f"modality_mask must be [B, C]. Got {modality_mask.shape}, "
                    f"expected {(B, C)}"
                )

        x_flat = x.reshape(B * C, 1, D, H, W)
        feat_flat = self.value_proj(x_flat)
        feat = feat_flat.reshape(B, C, self.output_channels, D, H, W)

        mod_emb = self.modality_embedding(modality_ids)

        gamma_beta = self.modality_to_gamma_beta(mod_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma[:, :, :, None, None, None]
        beta = beta[:, :, :, None, None, None]

        feat = feat * (1.0 + gamma) + beta

        mask = modality_mask[:, :, None, None, None, None]
        feat = feat * mask

        active_count = modality_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        if self.aggregation == "mean":
            y = feat.sum(dim=1) / active_count[:, :, None, None, None]
        elif self.aggregation == "sum":
            y = feat.sum(dim=1)
        elif self.aggregation == "sqrt_sum":
            y = feat.sum(dim=1) / torch.sqrt(active_count[:, :, None, None, None])
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        mod_emb_masked = mod_emb * modality_mask[:, :, None]
        modality_context = mod_emb_masked.sum(dim=1) / active_count

        return y, modality_context

class SpacingModalityContextFiLM(nn.Module):
    """
    FiLM conditioned on:
        log(spacing)
        log(anisotropy ratio)
        stage index
        pooled modality context
    """

    def __init__(
        self,
        channels: int,
        modality_embed_dim: int,
        hidden_dim: int = 128,
        stage_index: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.channels = channels
        self.modality_embed_dim = modality_embed_dim
        self.eps = eps

        self.register_buffer(
            "stage_index",
            torch.tensor([[float(stage_index)]], dtype=torch.float32),
        )

        input_dim = 5 + modality_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * channels),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_context: torch.Tensor,
    ) -> torch.Tensor:
        B, C, _, _, _ = x.shape

        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}")

        spacing = expand_spacing(spacing, B, x.device)

        modality_context = modality_context.to(device=x.device, dtype=torch.float32)

        if modality_context.shape != (B, self.modality_embed_dim):
            raise ValueError(
                f"modality_context must be [B, {self.modality_embed_dim}], "
                f"got {modality_context.shape}"
            )

        s_min = spacing.min(dim=1, keepdim=True).values
        s_max = spacing.max(dim=1, keepdim=True).values
        ratio = s_max / s_min.clamp_min(self.eps)

        spacing_features = torch.cat(
            [
                torch.log(spacing.clamp_min(self.eps)),
                torch.log(ratio.clamp_min(self.eps)),
                self.stage_index.to(x.device).expand(B, 1),
            ],
            dim=1,
        )

        condition = torch.cat([spacing_features, modality_context], dim=1)

        gamma_beta = self.net(condition).to(dtype=x.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]

        return x * (1.0 + gamma) + beta
