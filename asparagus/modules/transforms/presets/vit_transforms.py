from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from torchvision import transforms


def _spatial_axes(ndim: int) -> tuple[int, ...]:
    if ndim not in (2, 3):
        raise ValueError(
            f"Only 2D and 3D inputs are supported, received ndim={ndim}."
        )

    return tuple(range(ndim))


def CPU_vit_train_transforms(
    normalize: bool = True,
):
    """
    Full-volume segmentation preprocessing.
    """
    ndim = 3
    axes = _spatial_axes(ndim)

    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),

            # Mirroring preserves the full spatial extent and applies
            # consistently to the image and segmentation target.
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_vit_val_test_transforms(
    normalize: bool = True,
):
    """
    Full-volume segmentation validation preprocessing.

    No random transforms, padding, or cropping are applied.
    """
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
        ]
    )
