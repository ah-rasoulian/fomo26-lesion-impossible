from typing import Union, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Spacing, expand_spacing, to_3tuple, inverse_sigmoid


class PhysicalGaussianMaskedConv3d(nn.Module):
    """
    Spacing-aware 3D convolution.

    One shared kernel is learned, but its spatial support is softly masked
    according to physical spacing.

    x:       [B, C, H, W, D]
    spacing: [B, 3] or [3], order [H, W, D]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Optional[Union[int, Sequence[int]]] = None,
        dilation: Union[int, Sequence[int]] = 1,
        bias: bool = True,
        init_sigma_mm: float = 1.5,
        learn_sigma: bool = True,
        sigma_min_mm: float = 0.25,
        sigma_max_mm: float = 64.0,
        normalize_mask: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.dilation = to_3tuple(dilation)

        if any(k % 2 == 0 for k in self.kernel_size):
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        if padding is None:
            self.padding = tuple(k // 2 for k in self.kernel_size)
        else:
            self.padding = to_3tuple(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_mask = normalize_mask
        self.eps = eps

        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                *self.kernel_size,
            )
        )

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        init_sigma_mm = min(max(float(init_sigma_mm), sigma_min_mm), sigma_max_mm)
        p = (init_sigma_mm - sigma_min_mm) / (sigma_max_mm - sigma_min_mm)
        raw_init = torch.tensor(inverse_sigmoid(p), dtype=torch.float32)

        if learn_sigma:
            self.raw_sigma = nn.Parameter(raw_init)
        else:
            self.register_buffer("raw_sigma", raw_init)

        kh, kw, kd = self.kernel_size
        dy = torch.arange(-(kh // 2), kh // 2 + 1, dtype=torch.float32)
        dx = torch.arange(-(kw // 2), kw // 2 + 1, dtype=torch.float32)
        dz = torch.arange(-(kd // 2), kd // 2 + 1, dtype=torch.float32)

        yy, xx, zz = torch.meshgrid(dy, dx, dz, indexing="ij")
        offsets = torch.stack([yy, xx, zz], dim=-1).reshape(-1, 3)

        self.register_buffer("offsets", offsets)

    def sigma_mm(self) -> torch.Tensor:
        return self.sigma_min_mm + (
            self.sigma_max_mm - self.sigma_min_mm
        ) * torch.sigmoid(self.raw_sigma)

    def make_masks(self, spacing: torch.Tensor) -> torch.Tensor:
        """
        spacing: [B, 3]
        returns: [B, kH, kW, kD]
        """

        B = spacing.shape[0]
        kH, kW, kD = self.kernel_size

        spacing = spacing.to(device=self.weight.device, dtype=torch.float32)
        offsets = self.offsets.to(device=self.weight.device, dtype=torch.float32)

        sigma = self.sigma_mm().to(device=self.weight.device, dtype=torch.float32)
        sigma = sigma.clamp_min(self.eps)

        physical_offsets = offsets[None, :, :] * spacing[:, None, :]
        squared_distance = physical_offsets.square().sum(dim=-1)

        masks = torch.exp(-0.5 * squared_distance / sigma.square())
        masks = masks.view(B, kH, kW, kD)

        center_mask = torch.zeros_like(masks, dtype=torch.bool)
        center = (kH // 2, kW // 2, kD // 2)
        center_mask[:, center[0], center[1], center[2]] = True

        masks = torch.where(center_mask, torch.ones_like(masks), masks)

        if self.normalize_mask:
            target_mass = float(kH * kW * kD)
            current_mass = masks.square().sum(dim=(1, 2, 3), keepdim=True)
            gain = torch.sqrt(
                torch.tensor(target_mass, device=masks.device, dtype=masks.dtype)
                / current_mass.clamp_min(self.eps)
            )
            masks = masks * gain

        return masks.to(dtype=self.weight.dtype)

    def forward(self, x: torch.Tensor, spacing: Spacing) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, H, W, D], got {x.shape}")

        B, C, H, W, D = x.shape

        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        spacing = expand_spacing(spacing, B, x.device)
        masks = self.make_masks(spacing)

        weight_eff = self.weight[None, :, :, :, :, :] * masks[:, None, None, :, :, :]

        x_grouped = x.reshape(1, B * C, H, W, D)

        weight_grouped = weight_eff.reshape(
            B * self.out_channels,
            self.in_channels,
            *self.kernel_size,
        )

        bias_grouped = self.bias.repeat(B) if self.bias is not None else None

        y = F.conv3d(
            x_grouped,
            weight_grouped,
            bias=bias_grouped,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B,
        )

        _, _, H_out, W_out, D_out = y.shape
        y = y.reshape(B, self.out_channels, H_out, W_out, D_out)

        return y
