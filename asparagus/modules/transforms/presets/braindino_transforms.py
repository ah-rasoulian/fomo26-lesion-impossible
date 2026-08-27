from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from torchvision import transforms


SpatialShape = Tuple[int, int, int]


def _shape3(value: Sequence[int], name: str) -> SpatialShape:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item < 1 for item in result):
        raise ValueError(f"{name} must contain three positive integers.")
    return result


def _clone_metadata(info: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in info.items()
    }


@dataclass
class CenterCropByFraction3d:
    """Center-crop a subject volume while preserving its voxel spacing."""

    fraction: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1].")

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        result = dict(sample)
        image = result.get("image")
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("sample['image'] must have shape [C, H, W, D].")

        spatial_shape = tuple(int(size) for size in image.shape[-3:])
        crop_shape = tuple(
            max(1, min(size, int(math.floor(size * self.fraction))))
            for size in spatial_shape
        )
        starts = tuple(
            (size - crop_size) // 2
            for size, crop_size in zip(spatial_shape, crop_shape)
        )
        h, w, d = starts
        ch, cw, cd = crop_shape
        result["image"] = image[:, h : h + ch, w : w + cw, d : d + cd].contiguous()

        info = dict(result.get("info") or {})
        # Cropping changes the field of view, not the physical voxel size.
        info["valid_spatial_shapes"] = torch.tensor(
            crop_shape,
            dtype=torch.long,
        )
        result["info"] = info
        return result


@dataclass
class BrainDinoIntensityAugmentation:
    """MRI intensity augmentation sampled per subject and channel."""

    noise_probability: float = 0.2
    noise_std: float = 0.05
    gamma_probability: float = 0.2
    gamma_range: Tuple[float, float] = (0.8, 1.25)
    bias_probability: float = 0.2
    bias_strength: float = 0.25
    scale_shift_probability: float = 0.2
    scale_range: Tuple[float, float] = (0.9, 1.1)
    shift_range: Tuple[float, float] = (-0.1, 0.1)

    def __post_init__(self) -> None:
        for name, probability in (
            ("noise_probability", self.noise_probability),
            ("gamma_probability", self.gamma_probability),
            ("bias_probability", self.bias_probability),
            ("scale_shift_probability", self.scale_shift_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.noise_std < 0.0 or self.bias_strength < 0.0:
            raise ValueError("Noise and bias strengths must be non-negative.")
        for name, interval in (
            ("gamma_range", self.gamma_range),
            ("scale_range", self.scale_range),
            ("shift_range", self.shift_range),
        ):
            if len(interval) != 2 or interval[0] > interval[1]:
                raise ValueError(f"{name} must contain two ordered values.")
        if self.gamma_range[0] <= 0.0 or self.scale_range[0] <= 0.0:
            raise ValueError("Gamma and intensity-scale ranges must be positive.")

    def __call__(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected image with shape [B, C, H, W, D].")
        batch_channels = x.shape[:2]
        device, dtype = x.device, x.dtype

        if self.noise_probability > 0.0:
            apply = torch.rand(
                (*batch_channels, 1, 1, 1), device=device
            ) < self.noise_probability
            x = torch.where(apply, x + torch.randn_like(x) * self.noise_std, x)

        if self.gamma_probability > 0.0:
            apply = torch.rand(
                (*batch_channels, 1, 1, 1), device=device
            ) < self.gamma_probability
            gamma = torch.empty(
                (*batch_channels, 1, 1, 1), device=device, dtype=dtype
            ).uniform_(*self.gamma_range)
            minimum = x.amin(dim=(-3, -2, -1), keepdim=True)
            maximum = x.amax(dim=(-3, -2, -1), keepdim=True)
            value_range = (maximum - minimum).clamp_min(1e-6)
            normalized = ((x - minimum) / value_range).clamp(0.0, 1.0)
            x = torch.where(
                apply,
                normalized.pow(gamma) * value_range + minimum,
                x,
            )

        if self.bias_probability > 0.0:
            apply = torch.rand(
                (*batch_channels, 1, 1, 1), device=device
            ) < self.bias_probability
            axes = [
                torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
                for size in x.shape[-3:]
            ]
            hh, ww, dd = torch.meshgrid(*axes, indexing="ij")
            coefficients = torch.empty(
                (*batch_channels, 3), device=device, dtype=dtype
            ).uniform_(-self.bias_strength, self.bias_strength)
            field = (
                coefficients[..., 0, None, None, None] * hh
                + coefficients[..., 1, None, None, None] * ww
                + coefficients[..., 2, None, None, None] * dd
            ).exp()
            x = torch.where(apply, x * field, x)

        if self.scale_shift_probability > 0.0:
            apply = torch.rand(
                (*batch_channels, 1, 1, 1), device=device
            ) < self.scale_shift_probability
            scale = torch.empty(
                (*batch_channels, 1, 1, 1), device=device, dtype=dtype
            ).uniform_(*self.scale_range)
            shift = torch.empty(
                (*batch_channels, 1, 1, 1), device=device, dtype=dtype
            ).uniform_(*self.shift_range)
            x = torch.where(apply, x * scale + shift, x)

        if channel_mask is not None:
            channel_mask = torch.as_tensor(
                channel_mask, device=device, dtype=torch.bool
            )
            if channel_mask.shape != x.shape[:2]:
                raise ValueError("channel_mask must match image [B, C].")
            x = x * channel_mask[:, :, None, None, None].to(dtype=x.dtype)
        return x


class BrainDinoViewTransform:
    """
    Convert a padded full-volume batch into BrainDINO training views.

    Global views use each subject's complete valid volume. Batch padding is
    never augmented or treated as anatomy. A student/teacher global pair shares
    exactly the same spatial and intensity augmentation, preserving sparse
    iBOT token correspondence. Student channel/modality corruption is applied
    only after creation of the paired view.

    Local student views are optional. With ``local_crop_size=None``, requested
    local views are additional full-volume views; otherwise they are cropped
    independently and their valid spatial shapes are updated.
    """

    def __init__(
        self,
        token_grid_size: Sequence[int] = (32, 32, 32),
        local_crop_size: Optional[Sequence[int]] = None,
        n_global_crops: int = 2,
        n_local_crops: int = 0,
        mask_ratio: float = 0.5,
        training: bool = True,
        flip_probability: float = 0.0,
        flip_axes_hwd: Sequence[int] = (0, 1, 2),
        affine_probability: float = 0.4,
        rotation_degrees: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_fraction: float = 0.05,
        intensity_augmentation: Optional[BrainDinoIntensityAugmentation] = None,
        channel_drop_probability: float = 0.25,
        one_unknown_probability: float = 0.20,
        all_unknown_probability: float = 0.10,
        unknown_modality_id: int = 0,
        min_mask_block_fraction: float = 0.00025,
        max_mask_block_fraction: float = 0.002,
        local_center_jitter_fraction: float = 0.10,
        local_min_foreground_fraction: float = 0.05,
        local_crop_attempts: int = 8,
    ) -> None:
        self.token_grid_size = _shape3(token_grid_size, "token_grid_size")
        self.local_crop_size = (
            _shape3(local_crop_size, "local_crop_size")
            if local_crop_size is not None
            else None
        )
        if n_global_crops < 2:
            raise ValueError("DINO requires at least two global views.")
        if n_local_crops < 0:
            raise ValueError("n_local_crops cannot be negative.")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        for name, probability in (
            ("flip_probability", flip_probability),
            ("affine_probability", affine_probability),
            ("channel_drop_probability", channel_drop_probability),
            ("one_unknown_probability", one_unknown_probability),
            ("all_unknown_probability", all_unknown_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if one_unknown_probability + all_unknown_probability > 1.0:
            raise ValueError(
                "one_unknown_probability + all_unknown_probability "
                "must not exceed 1."
            )
        flip_axes = tuple(int(axis) for axis in flip_axes_hwd)
        if not flip_axes or len(set(flip_axes)) != len(flip_axes) or any(
            axis not in (0, 1, 2) for axis in flip_axes
        ):
            raise ValueError("flip_axes_hwd must contain unique H/W/D axes.")
        if rotation_degrees < 0.0:
            raise ValueError("rotation_degrees cannot be negative.")
        if (
            len(scale_range) != 2
            or scale_range[0] <= 0.0
            or scale_range[0] > scale_range[1]
        ):
            raise ValueError("scale_range must contain ordered positive values.")
        if not 0.0 <= translation_fraction <= 1.0:
            raise ValueError("translation_fraction must be in [0, 1].")
        if not 0.0 < min_mask_block_fraction <= max_mask_block_fraction <= 1.0:
            raise ValueError("Mask block fractions must satisfy 0 < min <= max <= 1.")
        if unknown_modality_id < 0:
            raise ValueError("unknown_modality_id must be non-negative.")
        if not 0.0 <= local_center_jitter_fraction <= 0.5:
            raise ValueError(
                "local_center_jitter_fraction must be between 0 and 0.5."
            )
        if not 0.0 <= local_min_foreground_fraction <= 1.0:
            raise ValueError(
                "local_min_foreground_fraction must be between 0 and 1."
            )
        if local_crop_attempts < 1:
            raise ValueError("local_crop_attempts must be positive.")

        self.n_global_crops = int(n_global_crops)
        self.n_local_crops = int(n_local_crops)
        self.mask_ratio = float(mask_ratio)
        self.training = bool(training)
        self.flip_probability = float(flip_probability)
        self.flip_axes_hwd = flip_axes
        self.affine_probability = float(affine_probability)
        self.rotation_degrees = float(rotation_degrees)
        self.scale_range = tuple(float(item) for item in scale_range)
        self.translation_fraction = float(translation_fraction)
        self.intensity_augmentation = (
            intensity_augmentation
            if intensity_augmentation is not None
            else BrainDinoIntensityAugmentation()
        )
        self.channel_drop_probability = float(channel_drop_probability)
        self.one_unknown_probability = float(one_unknown_probability)
        self.all_unknown_probability = float(all_unknown_probability)
        self.unknown_modality_id = int(unknown_modality_id)
        self.min_mask_block_fraction = float(min_mask_block_fraction)
        self.max_mask_block_fraction = float(max_mask_block_fraction)
        self.local_center_jitter_fraction = float(
            local_center_jitter_fraction
        )
        self.local_min_foreground_fraction = float(
            local_min_foreground_fraction
        )
        self.local_crop_attempts = int(local_crop_attempts)

    @staticmethod
    def _normalize_metadata(
        image: torch.Tensor,
        info: Mapping[str, Any],
        unknown_modality_id: int,
    ) -> Dict[str, torch.Tensor]:
        batch_size, channels = image.shape[:2]
        device = image.device
        if "spacing" not in info:
            raise KeyError("batch['info']['spacing'] is required.")
        spacing = torch.as_tensor(
            info["spacing"], device=device, dtype=torch.float32
        )
        if spacing.shape != (batch_size, 3):
            raise ValueError("info['spacing'] must be [B, 3].")
        if not torch.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Spacing values must be finite and positive.")

        modality_value = info.get("modality")
        modality = (
            torch.full(
                (batch_size, channels),
                unknown_modality_id,
                device=device,
                dtype=torch.long,
            )
            if modality_value is None
            else torch.as_tensor(modality_value, device=device, dtype=torch.long)
        )
        channel_mask_value = info.get("channel_mask")
        channel_mask = (
            torch.ones((batch_size, channels), device=device, dtype=torch.bool)
            if channel_mask_value is None
            else torch.as_tensor(
                channel_mask_value, device=device, dtype=torch.bool
            )
        )
        valid_value = info.get("valid_spatial_shapes")
        valid_shapes = (
            torch.tensor(
                image.shape[-3:], device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1).clone()
            if valid_value is None
            else torch.as_tensor(valid_value, device=device, dtype=torch.long)
        )
        if modality.shape != (batch_size, channels):
            raise ValueError("info['modality'] must be [B, C].")
        if channel_mask.shape != (batch_size, channels):
            raise ValueError("info['channel_mask'] must be [B, C].")
        if not channel_mask.any(dim=1).all():
            raise ValueError("Every subject must have at least one valid channel.")
        if valid_shapes.shape != (batch_size, 3):
            raise ValueError("info['valid_spatial_shapes'] must be [B, 3].")
        padded_shape = torch.tensor(
            image.shape[-3:], device=device, dtype=torch.long
        )
        if (valid_shapes < 1).any() or (valid_shapes > padded_shape).any():
            raise ValueError("A valid spatial shape exceeds the padded image.")
        return {
            "spacing": spacing,
            "modality": modality,
            "channel_mask": channel_mask,
            "valid_spatial_shapes": valid_shapes,
        }

    def _random_spatial_affine(
        self,
        x_hwd: torch.Tensor,
        spacing_hwd: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_hwd.shape[0]
        device = x_hwd.device
        apply = torch.rand(batch_size, device=device) < self.affine_probability
        if not apply.any():
            return x_hwd

        angles = torch.empty(
            batch_size, 3, device=device, dtype=torch.float32
        ).uniform_(
            -math.radians(self.rotation_degrees),
            math.radians(self.rotation_degrees),
        )
        scales = torch.empty(
            batch_size, device=device, dtype=torch.float32
        ).uniform_(*self.scale_range)
        translations = torch.empty(
            batch_size, 3, device=device, dtype=torch.float32
        ).uniform_(
            -2.0 * self.translation_fraction,
            2.0 * self.translation_fraction,
        )

        ax, ay, az = angles.unbind(dim=1)
        cx, cy, cz = ax.cos(), ay.cos(), az.cos()
        sx, sy, sz = ax.sin(), ay.sin(), az.sin()
        rotation_x = torch.zeros(batch_size, 3, 3, device=device)
        rotation_y = torch.zeros_like(rotation_x)
        rotation_z = torch.zeros_like(rotation_x)
        rotation_x[:, 0, 0] = 1.0
        rotation_x[:, 1, 1] = cx
        rotation_x[:, 1, 2] = -sx
        rotation_x[:, 2, 1] = sx
        rotation_x[:, 2, 2] = cx
        rotation_y[:, 0, 0] = cy
        rotation_y[:, 0, 2] = sy
        rotation_y[:, 1, 1] = 1.0
        rotation_y[:, 2, 0] = -sy
        rotation_y[:, 2, 2] = cy
        rotation_z[:, 0, 0] = cz
        rotation_z[:, 0, 1] = -sz
        rotation_z[:, 1, 0] = sz
        rotation_z[:, 1, 1] = cz
        rotation_z[:, 2, 2] = 1.0
        rotation = rotation_z @ rotation_y @ rotation_x

        spacing_grid = spacing_hwd[:, (1, 0, 2)].float()
        size_grid = torch.tensor(
            (x_hwd.shape[3], x_hwd.shape[2], x_hwd.shape[4]),
            device=device,
            dtype=torch.float32,
        )
        half_extent = (spacing_grid * size_grid[None]).clamp_min(1e-6)
        physical_to_normalized = torch.diag_embed(half_extent.reciprocal())
        normalized_to_physical = torch.diag_embed(half_extent)
        linear = (
            physical_to_normalized @ rotation @ normalized_to_physical
        ) * scales[:, None, None]
        theta = torch.zeros(batch_size, 3, 4, device=device)
        theta[:, :, :3] = linear
        theta[:, :, 3] = translations
        identity = torch.eye(3, 4, device=device).unsqueeze(0)
        theta = torch.where(apply[:, None, None], theta, identity)

        input_dtype = x_hwd.dtype
        x_dhw = x_hwd.permute(0, 1, 4, 2, 3).float()
        grid = F.affine_grid(theta, size=x_dhw.shape, align_corners=False)
        transformed = F.grid_sample(
            x_dhw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return transformed.to(input_dtype).permute(0, 1, 3, 4, 2).contiguous()

    def _augment_same_shape(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return x * channel_mask[:, :, None, None, None].to(x.dtype)
        x = self._random_spatial_affine(x, spacing)
        if self.flip_probability > 0.0:
            apply_flip = (
                torch.rand(x.shape[0], device=x.device) < self.flip_probability
            )
            if apply_flip.any():
                choices = torch.randint(
                    len(self.flip_axes_hwd),
                    (x.shape[0],),
                    device=x.device,
                )
                for option, axis in enumerate(self.flip_axes_hwd):
                    selected = apply_flip & (choices == option)
                    if selected.any():
                        x[selected] = torch.flip(x[selected], dims=(axis + 2,))
        return self.intensity_augmentation(x, channel_mask).contiguous()

    def _augment_padded_view(
        self,
        image: torch.Tensor,
        info: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        output = torch.zeros_like(image)
        groups: Dict[SpatialShape, list[int]] = defaultdict(list)
        for index, row in enumerate(info["valid_spatial_shapes"].tolist()):
            groups[tuple(int(item) for item in row)].append(index)

        for shape, indices_list in groups.items():
            indices = torch.tensor(indices_list, device=image.device)
            height, width, depth = shape
            valid_image = image[
                indices, :, :height, :width, :depth
            ].contiguous()
            augmented = self._augment_same_shape(
                valid_image,
                info["spacing"][indices],
                info["channel_mask"][indices],
            )
            output[indices, :, :height, :width, :depth] = augmented
        return output

    def _make_local_base(
        self,
        image: torch.Tensor,
        valid_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.local_crop_size is None:
            return image.clone(), valid_shapes.clone()

        target = self.local_crop_size
        output = image.new_zeros((image.shape[0], image.shape[1], *target))
        output_shapes = torch.empty_like(valid_shapes)
        for index, row in enumerate(valid_shapes.tolist()):
            original = tuple(int(item) for item in row)
            crop_shape = tuple(
                min(size, wanted) for size, wanted in zip(original, target)
            )
            subject = image[index, :, : original[0], : original[1], : original[2]]
            background, foreground_threshold = self._foreground_reference(
                subject
            )

            center_start = tuple(
                (size - crop_size) // 2
                for size, crop_size in zip(original, crop_shape)
            )
            candidates = []
            if self.training:
                for _ in range(self.local_crop_attempts):
                    candidate = []
                    for size, crop_size, centered in zip(
                        original, crop_shape, center_start
                    ):
                        maximum = size - crop_size
                        jitter_std = (
                            self.local_center_jitter_fraction * maximum
                        )
                        jitter = int(
                            torch.round(
                                torch.randn((), device=image.device)
                                * jitter_std
                            ).item()
                        )
                        candidate.append(
                            min(max(centered + jitter, 0), maximum)
                        )
                    candidates.append(tuple(candidate))
            # Exact center is both deterministic validation behavior and the
            # safe fallback for an unlucky set of randomized candidates.
            candidates.append(center_start)

            starts = center_start
            best_fraction = -1.0
            for candidate in candidates:
                h, w, d = candidate
                ch, cw, cd = crop_shape
                fraction = self._foreground_fraction(
                    subject[:, h : h + ch, w : w + cw, d : d + cd],
                    background,
                    foreground_threshold,
                )
                if fraction > best_fraction:
                    starts = candidate
                    best_fraction = fraction
                if fraction >= self.local_min_foreground_fraction:
                    starts = candidate
                    break
            h, w, d = starts
            ch, cw, cd = crop_shape
            output[index, :, :ch, :cw, :cd] = image[
                index, :, h : h + ch, w : w + cw, d : d + cd
            ]
            output_shapes[index] = torch.tensor(
                crop_shape, device=image.device, dtype=torch.long
            )
        return output, output_shapes

    @staticmethod
    def _foreground_reference(
        subject: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-channel corner background and a robust signal threshold."""
        corners = torch.stack(
            [
                subject[:, h, w, d]
                for h in (0, subject.shape[1] - 1)
                for w in (0, subject.shape[2] - 1)
                for d in (0, subject.shape[3] - 1)
            ],
            dim=-1,
        )
        background = corners.median(dim=-1).values[:, None, None, None]
        channel_range = subject.amax(dim=(1, 2, 3)) - subject.amin(
            dim=(1, 2, 3)
        )
        maximum = channel_range.amax()
        threshold = torch.where(
            torch.isfinite(maximum) & (maximum > 1e-8),
            maximum * 1e-3,
            maximum.new_tensor(float("inf")),
        )
        return background, threshold

    @staticmethod
    def _foreground_fraction(
        crop: torch.Tensor,
        background: torch.Tensor,
        threshold: torch.Tensor,
    ) -> float:
        foreground = (crop - background).abs().amax(dim=0) > threshold
        return float(foreground.float().mean().item())

    def _augment_student_info(
        self,
        info: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        augmented = _clone_metadata(info)
        modality = augmented["modality"].clone()
        channel_mask = augmented["channel_mask"].clone()
        if self.training:
            for batch_index in range(channel_mask.shape[0]):
                valid = torch.where(channel_mask[batch_index])[0]
                if (
                    valid.numel() > 1
                    and torch.rand((), device=channel_mask.device)
                    < self.channel_drop_probability
                ):
                    selected = valid[
                        torch.randint(valid.numel(), (), device=valid.device)
                    ]
                    channel_mask[batch_index, selected] = False

                valid = torch.where(channel_mask[batch_index])[0]
                identity_choice = torch.rand((), device=channel_mask.device)
                if identity_choice < self.all_unknown_probability:
                    modality[batch_index, valid] = self.unknown_modality_id
                elif identity_choice < (
                    self.all_unknown_probability + self.one_unknown_probability
                ):
                    selected = valid[
                        torch.randint(valid.numel(), (), device=valid.device)
                    ]
                    modality[batch_index, selected] = self.unknown_modality_id
        augmented["modality"] = modality
        augmented["channel_mask"] = channel_mask
        return augmented

    @staticmethod
    def _apply_channel_mask(
        view: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if channel_mask.shape != view.shape[:2]:
            raise ValueError("channel_mask and view batch/channel shapes differ.")
        return (
            view
            * channel_mask[:, :, None, None, None].to(
                device=view.device, dtype=view.dtype
            )
        ).contiguous()

    @staticmethod
    def _make_3d_block_mask(
        grid_shape: SpatialShape,
        number_masked: int,
        device: torch.device,
        generator: torch.Generator,
        min_block_fraction: float,
        max_block_fraction: float,
        max_attempts: int = 200,
    ) -> torch.Tensor:
        number_of_tokens = math.prod(grid_shape)
        mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)
        min_volume = max(1, round(min_block_fraction * number_of_tokens))
        max_volume = max(min_volume, round(max_block_fraction * number_of_tokens))

        attempts = 0
        masked_count = 0
        while masked_count < number_masked and attempts < max_attempts:
            attempts += 1
            remaining = number_masked - masked_count
            low_volume = min(min_volume, remaining)
            high_volume = min(max_volume, remaining)
            target_volume = int(
                torch.randint(
                    low_volume,
                    high_volume + 1,
                    (1,),
                    generator=generator,
                    device=device,
                ).item()
            )
            side = max(1, round(target_volume ** (1.0 / 3.0)))
            block_shape = []
            remaining_volume = target_volume
            for axis, axis_size in enumerate(grid_shape):
                if axis < 2:
                    low = max(1, side // 2)
                    high = min(axis_size, max(low, side * 2))
                    value = int(
                        torch.randint(
                            low,
                            high + 1,
                            (1,),
                            generator=generator,
                            device=device,
                        ).item()
                    )
                    block_shape.append(value)
                    remaining_volume = max(1, math.ceil(remaining_volume / value))
                else:
                    block_shape.append(min(axis_size, remaining_volume))

            starts = [
                int(
                    torch.randint(
                        axis_size - block_size + 1,
                        (1,),
                        generator=generator,
                        device=device,
                    ).item()
                )
                for axis_size, block_size in zip(grid_shape, block_shape)
            ]
            h, w, d = starts
            bh, bw, bd = block_shape
            block = mask[h : h + bh, w : w + bw, d : d + bd]
            available = (~block).nonzero(as_tuple=False)
            if available.numel() == 0:
                continue
            number_to_add = min(remaining, available.shape[0])
            selected = available[
                torch.randperm(
                    available.shape[0], generator=generator, device=device
                )[:number_to_add]
            ]
            block[selected[:, 0], selected[:, 1], selected[:, 2]] = True
            masked_count += number_to_add

        if masked_count < number_masked:
            available = (~mask).flatten().nonzero(as_tuple=False).squeeze(1)
            selected = available[
                torch.randperm(
                    available.numel(), generator=generator, device=device
                )[: number_masked - masked_count]
            ]
            mask.flatten()[selected] = True
        return mask

    def _make_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_index: int,
    ) -> torch.Tensor:
        number_of_tokens = math.prod(self.token_grid_size)
        number_masked = min(
            number_of_tokens,
            max(0, round(self.mask_ratio * number_of_tokens)),
        )
        if number_masked == 0:
            return torch.zeros(
                batch_size, number_of_tokens, device=device, dtype=torch.bool
            )

        masks = []
        for sample_index in range(batch_size):
            generator = torch.Generator(device=device)
            if self.training:
                seed = int(
                    torch.randint(
                        0, 2**31 - 1, (1,), device=device
                    ).item()
                )
            else:
                seed = 12_345 + mask_index * batch_size + sample_index
            generator.manual_seed(seed)
            masks.append(
                self._make_3d_block_mask(
                    self.token_grid_size,
                    number_masked,
                    device,
                    generator,
                    self.min_mask_block_fraction,
                    self.max_mask_block_fraction,
                ).flatten()
            )
        return torch.stack(masks, dim=0)

    def __call__(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        image = batch["image"]
        if not isinstance(image, torch.Tensor) or image.ndim != 5:
            raise ValueError("batch['image'] must be [B, C, H, W, D].")
        if not image.is_floating_point():
            image = image.float()

        normalized_info = self._normalize_metadata(
            image,
            batch.get("info", {}),
            self.unknown_modality_id,
        )
        teacher_global_info = [
            _clone_metadata(normalized_info)
            for _ in range(self.n_global_crops)
        ]
        student_global_info = [
            self._augment_student_info(normalized_info)
            for _ in range(self.n_global_crops)
        ]
        paired_global_views = [
            self._augment_padded_view(image, normalized_info)
            for _ in range(self.n_global_crops)
        ]

        # Student and teacher share the image storage. Their modality and channel
        # corruption remain different through their respective metadata.
        teacher_global_crops = list(
            paired_global_views
        )

        global_crops = list(
            paired_global_views
        )

        local_crops = []
        local_info = []
        for _ in range(self.n_local_crops):
            local_base, local_shapes = self._make_local_base(
                image,
                normalized_info["valid_spatial_shapes"],
            )
            base_info = _clone_metadata(normalized_info)
            base_info["valid_spatial_shapes"] = local_shapes

            student_info = (self._augment_student_info(base_info))

            local_view = (
                self._augment_padded_view(local_base, base_info)
            )

            local_crops.append(local_view)
            local_info.append(student_info)

        global_masks = [
            self._make_mask(
                image.shape[0],
                image.device,
                view_index,
            )
            for view_index in range(
                self.n_global_crops
            )
        ]

        result = dict(batch)

        result.update(
            {
                "global_crops": global_crops,
                "teacher_global_crops": teacher_global_crops,
                "local_crops": local_crops,
                "global_masks": global_masks,
                "global_info": student_global_info,
                "teacher_global_info": teacher_global_info,
                "local_info": local_info,
            }
        )
        result.pop("image", None)
        result.pop("label", None)
        result.pop("info", None)
        return result


def braindino_GPU_train_transforms(
    token_grid_size: Sequence[int] = (32, 32, 32),
    local_crop_size: Optional[Sequence[int]] = None,
    mask_ratio: float = 0.5,
    n_global_crops: int = 2,
    n_local_crops: int = 0,
    min_mask_block_fraction: float = 0.00025,
    max_mask_block_fraction: float = 0.002,
    local_center_jitter_fraction: float = 0.10,
    local_min_foreground_fraction: float = 0.05,
    local_crop_attempts: int = 8,
    channel_drop_probability: float = 0.35,
    one_unknown_probability: float = 0.25,
    all_unknown_probability: float = 0.10,
    unknown_modality_id: int = 0,
    flip_probability: float = 0.0,
    flip_axes_hwd: Sequence[int] = (0, 1, 2),
    affine_probability: float = 0.4,
    rotation_degrees: float = 10.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    translation_fraction: float = 0.05,
) -> BrainDinoViewTransform:
    return BrainDinoViewTransform(
        token_grid_size=token_grid_size,
        local_crop_size=local_crop_size,
        n_global_crops=n_global_crops,
        n_local_crops=n_local_crops,
        min_mask_block_fraction=min_mask_block_fraction,
        max_mask_block_fraction=max_mask_block_fraction,
        local_center_jitter_fraction=local_center_jitter_fraction,
        local_min_foreground_fraction=local_min_foreground_fraction,
        local_crop_attempts=local_crop_attempts,
        mask_ratio=mask_ratio,
        channel_drop_probability=channel_drop_probability,
        one_unknown_probability=one_unknown_probability,
        all_unknown_probability=all_unknown_probability,
        unknown_modality_id=unknown_modality_id,
        flip_probability=flip_probability,
        flip_axes_hwd=flip_axes_hwd,
        affine_probability=affine_probability,
        rotation_degrees=rotation_degrees,
        scale_range=scale_range,
        translation_fraction=translation_fraction,
        training=True,
    )


def braindino_GPU_val_transforms(
    token_grid_size: Sequence[int] = (32, 32, 32),
    local_crop_size: Optional[Sequence[int]] = None,
    mask_ratio: float = 0.5,
    n_global_crops: int = 2,
    n_local_crops: int = 0,
    min_mask_block_fraction: float = 0.00025,
    max_mask_block_fraction: float = 0.002,
    local_center_jitter_fraction: float = 0.0,
    local_min_foreground_fraction: float = 0.05,
    local_crop_attempts: int = 1,
) -> BrainDinoViewTransform:
    return BrainDinoViewTransform(
        token_grid_size=token_grid_size,
        local_crop_size=local_crop_size,
        n_global_crops=n_global_crops,
        n_local_crops=n_local_crops,
        min_mask_block_fraction=min_mask_block_fraction,
        max_mask_block_fraction=max_mask_block_fraction,
        local_center_jitter_fraction=local_center_jitter_fraction,
        local_min_foreground_fraction=local_min_foreground_fraction,
        local_crop_attempts=local_crop_attempts,
        mask_ratio=mask_ratio,
        flip_probability=0.0,
        affine_probability=0.0,
        channel_drop_probability=0.0,
        one_unknown_probability=0.0,
        all_unknown_probability=0.0,
        training=False,
    )


def braindino_CPU_train_transforms(
    normalize: bool = True,
    center_crop_fraction: float = 0.8,
):
    """Center-crop and normalize the subject volume on CPU."""
    return transforms.Compose(
        [
            CenterCropByFraction3d(center_crop_fraction),
            Torch_Normalize(normalize=normalize),
        ]
    )


def braindino_CPU_val_transforms(
    normalize: bool = True,
    center_crop_fraction: float = 0.8,
):
    """Apply deterministic center cropping and validation normalization."""
    return transforms.Compose(
        [
            CenterCropByFraction3d(center_crop_fraction),
            Torch_Normalize(normalize=normalize),
        ]
    )
