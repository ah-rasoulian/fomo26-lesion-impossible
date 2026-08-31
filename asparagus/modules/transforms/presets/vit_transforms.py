from __future__ import annotations
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.deep_supervision import Torch_DownsampleSegForDS
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from .braindino_transforms import CenterCropByFraction3d
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from torchvision import transforms

def _label_key(sample: Mapping[str, Any]) -> str:
    if "label" in sample:
        return "label"
    if "SEG_label" in sample:
        return "SEG_label"
    raise KeyError("Segmentation transform requires label or SEG_label.")


def _integer_label(label: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(label.float()).all():
        raise FloatingPointError("Label contains non-finite values.")
    rounded = label.round()
    if not torch.allclose(label.float(), rounded.float(), atol=1e-4, rtol=0.0):
        raise ValueError("Segmentation labels must contain integer class IDs.")
    if (rounded < 0).any():
        raise ValueError("Segmentation class IDs must be non-negative.")
    return rounded.to(label.dtype)


@dataclass
class RandomSegmentationAffine3d:
    """Joint shape-preserving 3-D affine/flip for images and class maps."""

    probability: float = 0.3
    rotation_degrees: float = 5.0
    scale_min: float = 0.95
    scale_max: float = 1.05
    translation_fraction: float = 0.02
    flip_probability: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("probability", self.probability),
            ("flip_probability", self.flip_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.rotation_degrees < 0.0:
            raise ValueError("rotation_degrees must be non-negative.")
        if not 0.0 < self.scale_min <= self.scale_max:
            raise ValueError("Require 0 < scale_min <= scale_max.")
        if not 0.0 <= self.translation_fraction <= 0.5:
            raise ValueError("translation_fraction must be in [0, 0.5].")

    @staticmethod
    def _rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
        ax, ay, az = angles
        del ax, ay, az
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)
        one = torch.ones((), dtype=angles.dtype, device=angles.device)
        zero = torch.zeros((), dtype=angles.dtype, device=angles.device)
        rx = torch.stack(
            (
                torch.stack((one, zero, zero)),
                torch.stack((zero, cx, -sx)),
                torch.stack((zero, sx, cx)),
            )
        )
        ry = torch.stack(
            (
                torch.stack((cy, zero, sy)),
                torch.stack((zero, one, zero)),
                torch.stack((-sy, zero, cy)),
            )
        )
        rz = torch.stack(
            (
                torch.stack((cz, -sz, zero)),
                torch.stack((sz, cz, zero)),
                torch.stack((zero, zero, one)),
            )
        )
        return rz @ ry @ rx

    def _affine(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Project layout is [C, H, W, D]; grid_sample expects [N, C, D, H, W].
        image_dhw = image.permute(0, 3, 1, 2).unsqueeze(0).float()
        label_dhw = label.permute(0, 3, 1, 2).unsqueeze(0).float()

        maximum_angle = math.radians(self.rotation_degrees)
        angles = torch.empty(3, device=image.device).uniform_(
            -maximum_angle, maximum_angle
        )
        scale = torch.empty((), device=image.device).uniform_(
            self.scale_min, self.scale_max
        )
        translation = torch.empty(3, device=image.device).uniform_(
            -self.translation_fraction, self.translation_fraction
        )

        theta = torch.zeros((1, 3, 4), device=image.device)
        theta[0, :, :3] = self._rotation_matrix(angles) / scale
        theta[0, :, 3] = 2.0 * translation
        grid = F.affine_grid(theta, image_dhw.shape, align_corners=False)

        transformed_image = F.grid_sample(
            image_dhw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        transformed_label = F.grid_sample(
            label_dhw,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        image = transformed_image[0].permute(0, 2, 3, 1).to(image.dtype)
        label = transformed_label[0].permute(0, 2, 3, 1).round().to(label.dtype)
        metadata = {
            "theta": theta[0].detach().cpu(),
            "angles_radians": angles.detach().cpu(),
            "scale": float(scale.detach().cpu()),
            "translation_fraction": translation.detach().cpu(),
        }
        return image, label, metadata

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        image = torch.as_tensor(result["image"])
        label = torch.as_tensor(result[_label_key(result)])
        if image.ndim != 4 or label.ndim != 4:
            raise ValueError("image and label must be [C, H, W, D].")
        if image.shape[1:] != label.shape[1:]:
            raise ValueError("Image and label spatial shapes must match.")
        label = _integer_label(label)

        original_shape = tuple(int(value) for value in image.shape[1:])
        foreground_before = int((label > 0).sum())
        affine_metadata = None
        if torch.rand(()).item() < self.probability:
            # Both calls inside _affine use this exact same grid. MRI uses
            # trilinear interpolation; the class map uses nearest neighbour.
            image, label, affine_metadata = self._affine(image, label)

        flipped_dimensions: list[int] = []
        # flip_probability is an overall per-sample probability. Conditional
        # on flipping, independently choose axes but ensure at least one axis.
        if torch.rand(()).item() < self.flip_probability:
            flipped_dimensions = [
                dimension
                for dimension in (1, 2, 3)
                if torch.rand(()).item() < 0.5
            ]
            if not flipped_dimensions:
                flipped_dimensions = [int(torch.randint(1, 4, ()).item())]
            image = torch.flip(image, dims=flipped_dimensions)
            label = torch.flip(label, dims=flipped_dimensions)

        if tuple(image.shape[1:]) != original_shape:
            raise RuntimeError("Spatial augmentation changed the image shape.")
        if image.shape[1:] != label.shape[1:]:
            raise RuntimeError("Spatial augmentation desynchronized image/label.")
        label = _integer_label(label).contiguous()
        result["image"] = image.contiguous()
        result["label"] = label
        result["SEG_label"] = label
        transforms_applied = dict(result.get("transforms_applied") or {})
        transforms_applied["segmentation_spatial"] = {
            "affine": affine_metadata,
            "flipped_tensor_dimensions": flipped_dimensions,
            "foreground_voxels_before": foreground_before,
            "foreground_voxels_after": int((label > 0).sum()),
        }
        result["transforms_applied"] = transforms_applied
        info = dict(result.get("info") or {})
        info["valid_spatial_shapes"] = torch.tensor(
            original_shape, dtype=torch.long
        )
        result["info"] = info
        return result


def verify_joint_segmentation_augmentation(
    trials: int = 16,
    minimum_dice: float = 0.90,
) -> float:
    """Synthetic check that MRI channels and the label share one transform.

    The synthetic image channels exactly equal the foreground mask before the
    transform. MRI interpolation softens boundaries, so agreement is checked
    after a 0.5 threshold rather than requiring bitwise equality.
    """
    if trials < 1:
        raise ValueError("trials must be positive.")
    label = torch.zeros((1, 48, 48, 48), dtype=torch.float32)
    label[:, 14:34, 17:36, 12:32] = 1.0
    image = label.repeat(3, 1, 1, 1)
    transform = RandomSegmentationAffine3d(
        probability=1.0,
        rotation_degrees=8.0,
        scale_min=0.95,
        scale_max=1.05,
        translation_fraction=0.03,
        flip_probability=1.0,
    )

    scores = []
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        for _ in range(trials):
            transformed = transform(
                {
                    "image": image.clone(),
                    "label": label.clone(),
                    "info": {},
                    "transforms_applied": {},
                }
            )
            transformed_label = transformed["label"] > 0
            for channel in transformed["image"]:
                transformed_image = channel.unsqueeze(0) >= 0.5
                intersection = (transformed_image & transformed_label).sum()
                denominator = transformed_image.sum() + transformed_label.sum()
                score = float(
                    (2.0 * intersection.float() / denominator.clamp_min(1)).cpu()
                )
                scores.append(score)

    worst_score = min(scores)
    if worst_score < minimum_dice:
        raise AssertionError(
            "Image/label augmentation alignment check failed: "
            f"minimum Dice={worst_score:.4f} < {minimum_dice:.4f}."
        )
    return worst_score


class ValidateSegmentationSample:
    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        image = torch.as_tensor(result["image"])
        label = torch.as_tensor(result[_label_key(result)])
        if image.ndim != 4 or label.ndim != 4:
            raise ValueError("image and label must be [C, H, W, D].")
        if image.shape[1:] != label.shape[1:]:
            raise ValueError(
                f"Image shape {tuple(image.shape)} and label shape "
                f"{tuple(label.shape)} do not match."
            )
        if not torch.isfinite(image).all():
            raise FloatingPointError("Image contains non-finite values.")
        label = _integer_label(label).contiguous()
        result["label"] = label
        result["SEG_label"] = label
        info = dict(result.get("info") or {})
        info["valid_spatial_shapes"] = torch.tensor(
            image.shape[1:], dtype=torch.long
        )
        result["info"] = info
        return result


def CPU_vit_seg_train_transforms(
    normalize: bool = True,
    affine_probability: float = 0.3,
    rotation_degrees: float = 5.0,
    scale_range: tuple[float, float] = (0.95, 1.05),
    translation_fraction: float = 0.02,
    flip_probability: float = 0.5,
):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            RandomSegmentationAffine3d(
                probability=affine_probability,
                rotation_degrees=rotation_degrees,
                scale_min=float(scale_range[0]),
                scale_max=float(scale_range[1]),
                translation_fraction=translation_fraction,
                flip_probability=flip_probability,
            ),
            ValidateSegmentationSample(),
        ]
    )


def CPU_vit_seg_val_transforms(normalize: bool = True):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            ValidateSegmentationSample(),
        ]
    )

def CPU_vit_train_transforms(
    normalize: bool = True,
    target_size: tuple[int, int, int] = (128, 128, 128),
    center_crop_fraction: float = 1.0,
):
    """Full-volume classification/regression training transforms.

    Spatial transforms preserve the post-center-crop tensor dimensions.
    Subject-level classification/regression targets are not transformed.
    """
    if len(target_size) == 2:
        axes = (0, 1)
    elif len(target_size) == 3:
        axes = (0, 1, 2)
    else:
        raise ValueError(
            "target_size must describe a 2D or 3D image, "
            f"but received {target_size}."
        )

    return transforms.Compose(
        [
            CenterCropByFraction3d(
                fraction=center_crop_fraction,
            ),
            Torch_Normalize(
                normalize=normalize,
            ),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.25,
                p_rot_per_axis=0.25,
                x_rot_in_degrees=(-10.0, 10.0),
                y_rot_in_degrees=(-10.0, 10.0),
                z_rot_in_degrees=(-10.0, 10.0),
                p_scale_all_channel=0.25,
                scale_factor=(0.90, 1.10),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=0.25,
                p_mirror_per_axis=0.25,
                axes=axes,
            ),
        ]
    )


def CPU_vit_val_test_transforms(
    normalize: bool = True,
    center_crop_fraction: float = 1.0,
):
    return transforms.Compose(
        [
            CenterCropByFraction3d(
                fraction=center_crop_fraction,
            ),
            Torch_Normalize(
                normalize=normalize,
            ),
        ]
    )


def GPU_vit_all_train_transforms(
    ndim: int = 3,
    deep_supervision: bool = False,
):
    """MRI intensity and acquisition-artifact augmentation.

    Probabilities are increased conservatively. Each transform remains
    independently randomized so that a training image is not necessarily
    subjected to every corruption.
    """
    if ndim not in (2, 3):
        raise ValueError(
            f"ndim must be 2 or 3, but received {ndim}."
        )

    # Retain the axis convention expected by the existing artifact transforms.
    axes = (0, ndim)

    tforms = transforms.Compose(
        [
            # Scanner resolution and reconstruction variability.
            Torch_Blur(
                p_per_channel=0.05,
            ),
            Torch_SimulateLowres(
                p_per_channel=0.10,
                p_per_axis=0.25,
            ),

            # Smooth intensity non-uniformity.
            Torch_BiasField(
                p_per_channel=0.15,
            ),

            # Global nonlinear contrast variability.
            Torch_Gamma(
                p_all_channel=0.15,
            ),

            # MRI acquisition and reconstruction artifacts.
            Torch_MotionGhosting(
                p_per_channel=0.05,
                axes=axes,
            ),
            Torch_GibbsRinging(
                p_per_channel=0.05,
                axes=axes,
            ),

            # Signal-level variability.
            Torch_MultiplicativeNoise(
                p_per_channel=0.10,
            ),
            Torch_AdditiveNoise(
                p_per_channel=0.10,
            ),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(
            Torch_DownsampleSegForDS(
                deep_supervision=True,
            )
        )

    return tforms
