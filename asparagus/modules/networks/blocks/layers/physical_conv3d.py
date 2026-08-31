from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Int3 = Union[int, Sequence[int]]
Spacing = Union[torch.Tensor, Sequence[float]]
OffsetMode = Literal["center", "random"]
PaddingMode = Literal["constant", "replicate"]


def to_3tuple(value: Int3) -> Tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        if len(value) != 3:
            raise ValueError(f"Expected three values, got {value}.")
        result = tuple(int(item) for item in value)
    if any(item < 1 for item in result):
        raise ValueError(f"All values must be positive, got {result}.")
    return result


def expand_spacing(
    spacing: Spacing,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    spacing = torch.as_tensor(spacing, device=device, dtype=torch.float32)
    if spacing.shape == (3,):
        spacing = spacing.unsqueeze(0).expand(batch_size, -1)
    if spacing.shape != (batch_size, 3):
        raise ValueError(
            f"spacing must be [3] or [B, 3], got {tuple(spacing.shape)}."
        )
    if not torch.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("All spacing values must be finite and positive.")
    return spacing


def inverse_sigmoid(probability: float) -> float:
    probability = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    return float(torch.logit(torch.tensor(probability)).item())


@dataclass(frozen=True)
class ConvGeometry3d:
    stride: Tuple[int, int, int]
    padding: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    cropping: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


@dataclass
class PhysicalConv3dOutput:
    features: torch.Tensor
    grid_shape: Tuple[int, int, int]
    grid_spacing_mm: torch.Tensor
    stride_vox: torch.Tensor
    effective_kernel_size_vox: torch.Tensor
    padding_vox: torch.Tensor
    cropping_vox: torch.Tensor
    original_spatial_shapes: torch.Tensor
    processed_spatial_shapes: torch.Tensor
    interpolation_applied: torch.Tensor


class PhysicalGaussianMaskedConv3d(nn.Module):
    """
    Spacing-aware convolution with bounded input adaptation and fixed output.

    The valid input range is derived independently for each spatial axis from
    the target token count T and effective convolution kernel K:

        minimum input length = T
        maximum input length = T * K

    Within this range, the dynamically selected stride is in [1, K]. This
    prevents gaps between adjacent convolutional receptive fields. If an input
    dimension is outside the range, only that dimension is clamped to the
    nearest boundary using trilinear interpolation. Spacing is updated to keep
    the physical field of view unchanged:

        processed_spacing = original_spacing * original_size / processed_size

    The learned physical Gaussian convolution is then applied directly to the
    processed image. No pooling is used. A centered (or optionally randomized)
    crop/padding offset makes the convolution output exactly ``grid_size``.

    With kernel 8 and target 32, the accepted shape range is [32, 256] on each
    axis. Thus 16 is interpolated to 32 and 512 is interpolated to 256.

    Tensor and spacing axis order is [H, W, D]. Padded heterogeneous batches
    are supported with ``valid_spatial_shapes=[B, 3]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Int3 = 8,
        grid_size: Int3 = (32, 32, 32),
        dilation: Int3 = 1,
        bias: bool = True,
        init_sigma_mm: float = 8.0,
        learn_sigma: bool = True,
        sigma_min_mm: float = 0.5,
        sigma_max_mm: float = 96.0,
        normalize_mask: bool = True,
        offset_mode: OffsetMode = "center",
        padding_mode: PaddingMode = "replicate",
        eps: float = 1e-6,
        safe_output_channels_per_call: int = 128,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = to_3tuple(kernel_size)
        self.grid_size = to_3tuple(grid_size)
        self.dilation = to_3tuple(dilation)
        self.normalize_mask = bool(normalize_mask)
        self.offset_mode = offset_mode
        self.padding_mode = padding_mode
        self.eps = float(eps)
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)
        # Runtime-only execution setting. It is deliberately not a parameter
        # or buffer, so old checkpoints retain exactly the same state_dict.
        self.safe_output_channels_per_call = int(
            safe_output_channels_per_call
        )

        if offset_mode not in ("center", "random"):
            raise ValueError(f"Unsupported offset_mode: {offset_mode}.")
        if padding_mode not in ("constant", "replicate"):
            raise ValueError(f"Unsupported padding_mode: {padding_mode}.")
        if not 0.0 < self.sigma_min_mm < self.sigma_max_mm:
            raise ValueError("Require 0 < sigma_min_mm < sigma_max_mm.")
        if self.safe_output_channels_per_call < 1:
            raise ValueError(
                "safe_output_channels_per_call must be positive."
            )

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                *self.kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        )
        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        init_sigma_mm = min(
            max(float(init_sigma_mm), self.sigma_min_mm),
            self.sigma_max_mm,
        )
        probability = (
            (init_sigma_mm - self.sigma_min_mm)
            / (self.sigma_max_mm - self.sigma_min_mm)
        )
        raw_sigma = torch.tensor(
            inverse_sigmoid(probability),
            dtype=torch.float32,
        )
        if learn_sigma:
            self.raw_sigma = nn.Parameter(raw_sigma)
        else:
            self.register_buffer("raw_sigma", raw_sigma)

        # Even kernels are centered between their central voxels; odd kernels
        # retain the usual zero-valued central offset.
        axes = [
            torch.arange(kernel, dtype=torch.float32) - (kernel - 1) / 2.0
            for kernel in self.kernel_size
        ]
        yy, xx, zz = torch.meshgrid(*axes, indexing="ij")
        offsets = torch.stack((yy, xx, zz), dim=-1).reshape(-1, 3)
        self.register_buffer("offsets", offsets)

    def sigma_mm(self) -> torch.Tensor:
        return self.sigma_min_mm + (
            self.sigma_max_mm - self.sigma_min_mm
        ) * torch.sigmoid(self.raw_sigma)

    def effective_kernel_size(self) -> Tuple[int, int, int]:
        return tuple(
            dilation * (kernel - 1) + 1
            for kernel, dilation in zip(self.kernel_size, self.dilation)
        )

    def input_shape_range(
        self,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        minimum = self.grid_size
        maximum = tuple(
            target * kernel
            for target, kernel in zip(
                self.grid_size,
                self.effective_kernel_size(),
            )
        )
        return minimum, maximum

    def clamp_input_shape(
        self,
        input_shape: Sequence[int],
    ) -> Tuple[int, int, int]:
        input_shape = to_3tuple(input_shape)
        minimum, maximum = self.input_shape_range()
        return tuple(
            min(max(length, lower), upper)
            for length, lower, upper in zip(input_shape, minimum, maximum)
        )

    def make_masks(self, spacing: torch.Tensor) -> torch.Tensor:
        """Construct one physical Gaussian kernel mask per batch item."""

        batch_size = spacing.shape[0]
        spacing = spacing.to(device=self.weight.device, dtype=torch.float32)
        offsets = self.offsets.to(
            device=self.weight.device,
            dtype=torch.float32,
        )
        sigma = self.sigma_mm().to(
            device=self.weight.device,
            dtype=torch.float32,
        ).clamp_min(self.eps)

        physical_offsets = offsets.unsqueeze(0) * spacing.unsqueeze(1)
        squared_distance = physical_offsets.square().sum(dim=-1)
        masks = torch.exp(-0.5 * squared_distance / sigma.square())
        masks = masks.reshape(batch_size, *self.kernel_size)

        if self.normalize_mask:
            target_mass = float(torch.tensor(self.kernel_size).prod().item())
            current_mass = masks.square().sum(
                dim=(1, 2, 3),
                keepdim=True,
            )
            masks = masks * torch.sqrt(
                masks.new_tensor(target_mass)
                / current_mass.clamp_min(self.eps)
            )
        return masks.to(dtype=self.weight.dtype)

    def _split_adjustment(self, total: int) -> Tuple[int, int]:
        if self.offset_mode == "random" and self.training and total > 0:
            left = int(
                torch.randint(0, total + 1, size=()).item()
            )
        else:
            left = total // 2
        return left, total - left

    def compute_geometry(
        self,
        processed_shape: Sequence[int],
    ) -> ConvGeometry3d:
        """Compute stride and centered crop/padding offset for an exact grid."""

        processed_shape = to_3tuple(processed_shape)
        minimum, maximum = self.input_shape_range()
        if any(
            length < lower or length > upper
            for length, lower, upper in zip(
                processed_shape,
                minimum,
                maximum,
            )
        ):
            raise ValueError(
                f"processed_shape {processed_shape} is outside the supported "
                f"range [{minimum}, {maximum}]."
            )

        strides = []
        padding_pairs = []
        cropping_pairs = []
        for length, kernel, target in zip(
            processed_shape,
            self.effective_kernel_size(),
            self.grid_size,
        ):
            if target == 1:
                stride = 1
            else:
                numerator = max(length - kernel, 0)
                denominator = target - 1
                # Nearest integer stride minimizes required crop/padding.
                stride = max(
                    1,
                    min(
                        kernel,
                        (numerator + denominator // 2) // denominator,
                    ),
                )

            required_extent = (target - 1) * stride + kernel
            adjustment = required_extent - length
            if adjustment >= 0:
                padding = self._split_adjustment(adjustment)
                cropping = (0, 0)
            else:
                padding = (0, 0)
                cropping = self._split_adjustment(-adjustment)

            strides.append(stride)
            padding_pairs.append(padding)
            cropping_pairs.append(cropping)

        return ConvGeometry3d(
            stride=tuple(strides),
            padding=tuple(padding_pairs),  # type: ignore[arg-type]
            cropping=tuple(cropping_pairs),  # type: ignore[arg-type]
        )

    def _resize_if_needed(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        original_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int], bool]:
        processed_shape = self.clamp_input_shape(original_shape)
        interpolated = processed_shape != original_shape
        if interpolated:
            x = F.interpolate(
                x,
                size=processed_shape,
                mode="trilinear",
                align_corners=False,
            )
            original_size = spacing.new_tensor(original_shape)
            processed_size = spacing.new_tensor(processed_shape)
            spacing = spacing * original_size / processed_size
        return x, spacing, processed_shape, interpolated

    def _crop_and_pad(
        self,
        x: torch.Tensor,
        geometry: ConvGeometry3d,
    ) -> torch.Tensor:
        (crop_h, crop_w, crop_d) = geometry.cropping
        height_end = x.shape[2] - crop_h[1]
        width_end = x.shape[3] - crop_w[1]
        depth_end = x.shape[4] - crop_d[1]
        x = x[
            :,
            :,
            crop_h[0] : height_end,
            crop_w[0] : width_end,
            crop_d[0] : depth_end,
        ]

        pad_h, pad_w, pad_d = geometry.padding
        pad_flat = (
            pad_d[0],
            pad_d[1],
            pad_w[0],
            pad_w[1],
            pad_h[0],
            pad_h[1],
        )
        if any(pad_flat):
            if self.padding_mode == "constant":
                x = F.pad(x, pad_flat, mode="constant", value=0.0)
            else:
                x = F.pad(x, pad_flat, mode="replicate")
        return x

    def _grouped_convolution(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        geometry: ConvGeometry3d,
    ) -> torch.Tensor:
        """Apply exactly the original masked convolution with safer dispatch.

        When all rows share spacing, all effective kernels are identical. In
        that common case, keeping samples in the real batch dimension is both
        faster and more cuDNN-friendly than converting the batch into groups.

        A single large volume cannot be split along its batch dimension by
        older cuDNN versions. For that case only, split the output filters into
        bounded chunks and concatenate their outputs. Every chunk uses the
        same input, stride, dilation, Gaussian mask, weights, and bias as the
        original convolution, so this does not change the model function.
        """
        batch_size = x.shape[0]
        x = self._crop_and_pad(x, geometry)

        shared_spacing = batch_size == 1 or torch.equal(
            spacing,
            spacing[0:1].expand_as(spacing),
        )
        if shared_spacing:
            y = self._shared_kernel_convolution(
                x=x,
                spacing=spacing[0:1],
                geometry=geometry,
                split_output_channels=(batch_size == 1),
            )
        else:
            y = self._per_sample_kernel_convolution(
                x=x,
                spacing=spacing,
                geometry=geometry,
            )

        actual_grid = tuple(int(item) for item in y.shape[2:])
        if actual_grid != self.grid_size:
            raise RuntimeError(
                f"Expected convolution grid {self.grid_size}, got {actual_grid}."
            )
        return y

    def _output_channel_slices(
        self,
        split: bool,
    ) -> list[Tuple[int, int]]:
        step = (
            min(self.out_channels, self.safe_output_channels_per_call)
            if split
            else self.out_channels
        )
        return [
            (start, min(start + step, self.out_channels))
            for start in range(0, self.out_channels, step)
        ]

    def _shared_kernel_convolution(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        geometry: ConvGeometry3d,
        split_output_channels: bool,
    ) -> torch.Tensor:
        """Fast path: one effective kernel shared across the real batch."""
        mask = self.make_masks(spacing)[0]
        outputs = []
        for start, end in self._output_channel_slices(
            split_output_channels
        ):
            weight = (
                self.weight[start:end]
                * mask[None, None, :, :, :]
            )
            bias = (
                self.bias[start:end]
                if self.bias is not None
                else None
            )
            outputs.append(
                F.conv3d(
                    x,
                    weight,
                    bias=bias,
                    stride=geometry.stride,
                    padding=0,
                    dilation=self.dilation,
                    groups=1,
                )
            )
        return torch.cat(outputs, dim=1)

    def _per_sample_kernel_convolution(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        geometry: ConvGeometry3d,
    ) -> torch.Tensor:
        """General path for different physical kernels in one shape group."""
        batch_size = x.shape[0]
        masks = self.make_masks(spacing)
        x_grouped = x.reshape(
            1,
            batch_size * self.in_channels,
            *x.shape[2:],
        )

        outputs = []
        # The grouped representation always has pseudo-batch size one, so use
        # bounded output chunks to avoid large non-batch-splittable cuDNN ops.
        for start, end in self._output_channel_slices(split=True):
            chunk_channels = end - start
            weight = (
                self.weight[start:end].unsqueeze(0)
                * masks[:, None, None, :, :, :]
            ).reshape(
                batch_size * chunk_channels,
                self.in_channels,
                *self.kernel_size,
            )
            bias = (
                self.bias[start:end].repeat(batch_size)
                if self.bias is not None
                else None
            )
            output = F.conv3d(
                x_grouped,
                weight,
                bias=bias,
                stride=geometry.stride,
                padding=0,
                dilation=self.dilation,
                groups=batch_size,
            )
            outputs.append(
                output.reshape(
                    batch_size,
                    chunk_channels,
                    *output.shape[2:],
                )
            )
        return torch.cat(outputs, dim=1)

    def _prepare_valid_shapes(
        self,
        x: torch.Tensor,
        valid_spatial_shapes: Optional[
            Union[torch.Tensor, Sequence[Sequence[int]]]
        ],
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        padded_shape = torch.tensor(x.shape[2:], dtype=torch.long)
        if valid_spatial_shapes is None:
            return padded_shape.unsqueeze(0).expand(batch_size, -1).clone()

        shapes = torch.as_tensor(
            valid_spatial_shapes,
            dtype=torch.long,
            device="cpu",
        )
        if shapes.shape != (batch_size, 3):
            raise ValueError(
                "valid_spatial_shapes must be [B, 3], got "
                f"{tuple(shapes.shape)}."
            )
        if (shapes < 1).any() or (shapes > padded_shape).any():
            raise ValueError(
                "Every valid shape must be positive and no larger than "
                f"the padded tensor shape {tuple(x.shape[2:])}."
            )
        return shapes

    def _forward_same_shape(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        original_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int], ConvGeometry3d, bool]:
        x, processed_spacing, processed_shape, interpolated = (
            self._resize_if_needed(x, spacing, original_shape)
        )
        geometry = self.compute_geometry(processed_shape)
        features = self._grouped_convolution(
            x,
            processed_spacing,
            geometry,
        )
        return (
            features,
            processed_spacing,
            processed_shape,
            geometry,
            interpolated,
        )

    def forward(
        self,
        x: torch.Tensor,
        spacing: Spacing,
        valid_spatial_shapes: Optional[
            Union[torch.Tensor, Sequence[Sequence[int]]]
        ] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, PhysicalConv3dOutput]:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, H, W, D], got {x.shape}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}."
            )

        batch_size = x.shape[0]
        spacing = expand_spacing(spacing, batch_size, x.device)
        original_shapes = self._prepare_valid_shapes(x, valid_spatial_shapes)

        if valid_spatial_shapes is None:
            original_shape = tuple(int(item) for item in x.shape[2:])
            (
                features,
                processed_spacing,
                processed_shape,
                geometry,
                interpolated,
            ) = self._forward_same_shape(x, spacing, original_shape)
            if not return_metadata:
                return features

            stride = torch.tensor(
                geometry.stride,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
            padding = torch.tensor(
                geometry.padding,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            cropping = torch.tensor(
                geometry.cropping,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            processed_shapes = torch.tensor(
                processed_shape,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
            return PhysicalConv3dOutput(
                features=features,
                grid_shape=self.grid_size,
                grid_spacing_mm=(
                    processed_spacing * stride.to(processed_spacing.dtype)
                ),
                stride_vox=stride,
                effective_kernel_size_vox=torch.tensor(
                    self.effective_kernel_size(),
                    device=x.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1),
                padding_vox=padding,
                cropping_vox=cropping,
                original_spatial_shapes=original_shapes.to(x.device),
                processed_spatial_shapes=processed_shapes,
                interpolation_applied=torch.full(
                    (batch_size,),
                    interpolated,
                    device=x.device,
                    dtype=torch.bool,
                ),
            )

        shape_groups: dict[Tuple[int, int, int], list[int]] = {}
        for index, shape_row in enumerate(original_shapes.tolist()):
            shape = tuple(int(item) for item in shape_row)
            shape_groups.setdefault(shape, []).append(index)

        grouped_outputs = []
        grouped_indices = []
        processed_spacing_rows = torch.empty_like(spacing)
        stride_rows = torch.empty(batch_size, 3, device=x.device, dtype=torch.long)
        padding_rows = torch.empty(batch_size, 3, 2, device=x.device, dtype=torch.long)
        cropping_rows = torch.empty(batch_size, 3, 2, device=x.device, dtype=torch.long)
        processed_shape_rows = torch.empty(batch_size, 3, device=x.device, dtype=torch.long)
        interpolation_rows = torch.empty(batch_size, device=x.device, dtype=torch.bool)

        for original_shape, indices in shape_groups.items():
            index_tensor = torch.tensor(indices, device=x.device, dtype=torch.long)
            height, width, depth = original_shape
            # Slice the padded spatial tensor before gathering batch entries.
            # index_select materializes its full input shape, so gathering
            # first can allocate several GiB when one subject determines a
            # very large heterogeneous batch canvas.
            samples = x[:, :, :height, :width, :depth].index_select(
                0, index_tensor
            )
            sample_spacing = spacing.index_select(0, index_tensor)
            (
                output,
                processed_spacing,
                processed_shape,
                geometry,
                interpolated,
            ) = self._forward_same_shape(
                samples,
                sample_spacing,
                original_shape,
            )
            grouped_outputs.append(output)
            grouped_indices.append(index_tensor)
            processed_spacing_rows[index_tensor] = processed_spacing
            stride_rows[index_tensor] = torch.tensor(
                geometry.stride, device=x.device, dtype=torch.long
            )
            padding_rows[index_tensor] = torch.tensor(
                geometry.padding, device=x.device, dtype=torch.long
            )
            cropping_rows[index_tensor] = torch.tensor(
                geometry.cropping, device=x.device, dtype=torch.long
            )
            processed_shape_rows[index_tensor] = torch.tensor(
                processed_shape, device=x.device, dtype=torch.long
            )
            interpolation_rows[index_tensor] = interpolated

        outputs_grouped = torch.cat(grouped_outputs, dim=0)
        original_indices = torch.cat(grouped_indices, dim=0)
        restore_order = torch.argsort(original_indices)
        features = outputs_grouped.index_select(0, restore_order)
        if not return_metadata:
            return features

        return PhysicalConv3dOutput(
            features=features,
            grid_shape=self.grid_size,
            grid_spacing_mm=(
                processed_spacing_rows
                * stride_rows.to(processed_spacing_rows.dtype)
            ),
            stride_vox=stride_rows,
            effective_kernel_size_vox=torch.tensor(
                self.effective_kernel_size(),
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1),
            padding_vox=padding_rows,
            cropping_vox=cropping_rows,
            original_spatial_shapes=original_shapes.to(x.device),
            processed_spatial_shapes=processed_shape_rows,
            interpolation_applied=interpolation_rows,
        )


def _smoke_test() -> None:
    # Small equivalent of kernel=8, target=32: valid range [4, 8].
    layer = PhysicalGaussianMaskedConv3d(
        in_channels=1,
        out_channels=6,
        kernel_size=2,
        grid_size=(4, 4, 4),
        offset_mode="center",
    ).eval()
    assert layer.input_shape_range() == ((4, 4, 4), (8, 8, 8))

    padded = torch.randn(2, 1, 16, 8, 8)
    spacing = torch.tensor(((2.0, 1.0, 1.0), (0.5, 1.0, 1.0)))
    with torch.inference_mode():
        output = layer(
            padded,
            spacing,
            valid_spatial_shapes=((2, 8, 8), (16, 8, 8)),
            return_metadata=True,
        )

    assert output.features.shape == (2, 6, 4, 4, 4)
    assert output.processed_spatial_shapes.tolist() == [[4, 8, 8], [8, 8, 8]]
    assert torch.allclose(output.grid_spacing_mm[0, 0], torch.tensor(1.0))
    assert torch.allclose(output.grid_spacing_mm[1, 0], torch.tensor(2.0))
    assert output.interpolation_applied.tolist() == [True, True]

    # Verify the shared-kernel chunked path against the original grouped
    # formulation and confirm that no new state_dict entries were introduced.
    chunked = PhysicalGaussianMaskedConv3d(
        in_channels=1,
        out_channels=5,
        kernel_size=2,
        grid_size=(4, 4, 4),
        offset_mode="center",
        safe_output_channels_per_call=2,
    ).eval()
    state_keys = set(chunked.state_dict())
    assert state_keys == {"weight", "bias", "raw_sigma", "offsets"}

    image = torch.randn(1, 1, 8, 8, 8)
    shared_spacing = torch.ones(1, 3)
    geometry = chunked.compute_geometry((8, 8, 8))
    optimized = chunked._grouped_convolution(
        image,
        shared_spacing,
        geometry,
    )

    mask = chunked.make_masks(shared_spacing)
    effective_weight = (
        chunked.weight.unsqueeze(0)
        * mask[:, None, None, :, :, :]
    ).reshape(5, 1, 2, 2, 2)
    reference = F.conv3d(
        chunked._crop_and_pad(image, geometry),
        effective_weight,
        bias=chunked.bias,
        stride=geometry.stride,
    )
    assert torch.allclose(optimized, reference, atol=1e-6, rtol=1e-5)

    clone = PhysicalGaussianMaskedConv3d(
        in_channels=1,
        out_channels=5,
        kernel_size=2,
        grid_size=(4, 4, 4),
        safe_output_channels_per_call=3,
    )
    clone.load_state_dict(chunked.state_dict(), strict=True)


if __name__ == "__main__":
    _smoke_test()
    print("PhysicalGaussianMaskedConv3d smoke test passed")
