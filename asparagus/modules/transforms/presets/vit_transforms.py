from asparagus.modules.transforms.pad import Torch_Pad
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from torchvision import transforms


def CPU_vit_train_transforms(
    normalize: bool = True,
    target_size: tuple[int, int, int] = (128, 128, 128),
):
    """
    Full-volume segmentation preprocessing.
    """
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_vit_val_test_transforms(
    normalize: bool = True,
    target_size: tuple[int, int, int] = (128, 128, 128),
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
