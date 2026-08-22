from __future__ import annotations
from typing import Any, Sequence
import torch

from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.deep_supervision import Torch_DownsampleSegForDS
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from asparagus.modules.transforms.crop import Torch_Crop
from asparagus.modules.transforms.pad import Torch_Pad
from torchvision import transforms


class Torch_PadToDivisible:
    """Pad every spatial dimension to a multiple of `divisor`."""

    def __init__(
        self,
        divisor: int = 8,
        data_key: str = "image",
        label_key: str = "label",
        pad_value: str | int | float = "min",
    ):
        if divisor <= 0:
            raise ValueError("divisor must be positive.")

        self.divisor = int(divisor)
        self.data_key = data_key
        self.label_key = label_key
        self.pad_value = pad_value

    def __call__(self, data_dict: dict) -> dict:
        image = data_dict[self.data_key]

        # Image shape is [C, D, H, W] or [C, H, W].
        spatial_shape = image.shape[1:]

        padded_shape = tuple(
            ((int(size) + self.divisor - 1) // self.divisor)
            * self.divisor
            for size in spatial_shape
        )

        if tuple(spatial_shape) == padded_shape:
            return data_dict

        return Torch_Pad(
            data_key=self.data_key,
            label_key=self.label_key,
            patch_size=padded_shape,
            pad_value=self.pad_value,
        )(data_dict)

class Torch_RandomSizeCrop:
    """Randomly crop to a size between `min_size` and `max_size`.

    A common random scale is applied to every spatial dimension, preserving
    the aspect ratio implied by `min_size`. The selected dimensions are rounded
    to multiples of `size_divisor`, which should normally match the ViT patch
    size.

    Padding is performed only when cropping is selected and only to ensure that
    the requested crop fits inside the input image.
    """

    def __init__(
        self,
        min_size: Sequence[int],
        max_size: Sequence[int] | None = None,
        p: float = 0.5,
        size_divisor: int = 8,
        p_oversample_foreground: float = 0.0,
    ) -> None:
        self.min_size = tuple(int(size) for size in min_size)

        if max_size is None:
            self.max_size = tuple(
                2 * size for size in self.min_size
            )
        else:
            self.max_size = tuple(
                int(size) for size in max_size
            )

        if not self.min_size:
            raise ValueError("min_size cannot be empty.")

        if len(self.min_size) != len(self.max_size):
            raise ValueError(
                "min_size and max_size must have the same number "
                "of dimensions."
            )

        if any(size <= 0 for size in self.min_size):
            raise ValueError(
                "Every min_size dimension must be positive."
            )

        if any(size <= 0 for size in self.max_size):
            raise ValueError(
                "Every max_size dimension must be positive."
            )

        if any(
            maximum < minimum
            for minimum, maximum in zip(
                self.min_size,
                self.max_size,
            )
        ):
            raise ValueError(
                "Every max_size dimension must be greater than or "
                "equal to the corresponding min_size dimension."
            )

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")

        if size_divisor <= 0:
            raise ValueError(
                "size_divisor must be positive."
            )

        if not 0.0 <= p_oversample_foreground <= 1.0:
            raise ValueError(
                "p_oversample_foreground must be in [0, 1]."
            )

        self.p = float(p)
        self.size_divisor = int(size_divisor)
        self.p_oversample_foreground = float(
            p_oversample_foreground
        )

        scale_limits = [
            maximum / minimum
            for minimum, maximum in zip(
                self.min_size,
                self.max_size,
            )
        ]
        self.maximum_common_scale = min(scale_limits)

    def _round_to_divisor(
        self,
        size: float,
        minimum: int,
        maximum: int,
    ) -> int:
        rounded = int(
            round(size / self.size_divisor)
            * self.size_divisor
        )

        rounded = max(minimum, rounded)
        rounded = min(maximum, rounded)

        return rounded

    def _sample_crop_size(self) -> tuple[int, ...]:
        random_scale = float(
            torch.empty(1).uniform_(
                1.0,
                self.maximum_common_scale,
            )
        )

        return tuple(
            self._round_to_divisor(
                size=minimum * random_scale,
                minimum=minimum,
                maximum=maximum,
            )
            for minimum, maximum in zip(
                self.min_size,
                self.max_size,
            )
        )

    def __call__(self, sample: Any) -> Any:
        if self.p == 0.0:
            return sample

        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            return sample

        crop_size = self._sample_crop_size()

        # Padding and cropping are kept together so samples for which the
        # transform is skipped retain their original full-volume size.
        sample = Torch_Pad(
            patch_size=crop_size,
        )(sample)

        sample = Torch_Crop(
            patch_size=crop_size,
            p_oversample_foreground=(
                self.p_oversample_foreground
            ),
        )(sample)

        return sample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"min_size={self.min_size}, "
            f"max_size={self.max_size}, "
            f"p={self.p}, "
            f"size_divisor={self.size_divisor}, "
            "p_oversample_foreground="
            f"{self.p_oversample_foreground})"
        )

def CPU_vit_train_transforms(
    normalize: bool = True,
    target_size: tuple[int, int, int] = (128, 128, 128),
):
    """Full-volume classification/regression training preprocessing.

    Spatial augmentations are applied to the image only because classification
    and regression labels are subject-level targets.
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
            Torch_Normalize(
                normalize=normalize,
            ),
            Torch_RandomSizeCrop(
                min_size=target_size,
                max_size=tuple(2 * size for size in target_size),
                p=0.5,
                size_divisor=8,
                p_oversample_foreground=0.0,
            ),
            Torch_Spatial(
                patch_size=target_size,

                # Mild deformation. Avoid strong deformation because small
                # infarcts should not be erased or severely distorted.
                p_deform_all_channel=0.10,

                # More frequent but less extreme rotations.
                p_rot_all_channel=0.50,
                p_rot_per_axis=0.50,
                x_rot_in_degrees=(-15.0, 15.0),
                y_rot_in_degrees=(-15.0, 15.0),
                z_rot_in_degrees=(-15.0, 15.0),

                # Frequent, anatomically plausible scale augmentation.
                p_scale_all_channel=0.40,
                scale_factor=(0.85, 1.20),

                # Preserve the full-volume pipeline.
                crop=False,
                clip_to_input_range=False,

                # Classification/regression labels are scalar targets and
                # must not be spatially transformed.
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.50,
                axes=axes,
            ),
            Torch_PadToDivisible(
                divisor=8,
                pad_value="min",
            ),
        ]
    )


def CPU_vit_val_test_transforms(
    normalize: bool = True,
    target_size: tuple[int, int, int] = (128, 128, 128),
):
    """Deterministic full-volume validation and test preprocessing."""
    return transforms.Compose(
        [
            Torch_Normalize(
                normalize=normalize,
            ),
            Torch_PadToDivisible(
                divisor=8,
                pad_value="min",
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
                p_per_channel=0.25,
            ),
            Torch_SimulateLowres(
                p_per_channel=0.40,
                p_per_axis=0.35,
            ),

            # Smooth intensity non-uniformity.
            Torch_BiasField(
                p_per_channel=0.30,
            ),

            # Global nonlinear contrast variability.
            Torch_Gamma(
                p_all_channel=0.30,
            ),

            # MRI acquisition and reconstruction artifacts.
            Torch_MotionGhosting(
                p_per_channel=0.20,
                axes=axes,
            ),
            Torch_GibbsRinging(
                p_per_channel=0.20,
                axes=axes,
            ),

            # Signal-level variability.
            Torch_MultiplicativeNoise(
                p_per_channel=0.20,
            ),
            Torch_AdditiveNoise(
                p_per_channel=0.20,
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