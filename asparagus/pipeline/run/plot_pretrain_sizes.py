import hydra
import nibabel as nib
import numpy as np
import random
import torch
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.hydra.plugins.searchpath_plugins import (
    PretrainSearchpathPlugin,
)
from asparagus.paths import get_config_path
from dotenv import load_dotenv
from gardening_tools.functional.paths.read import load_json
from hydra.core.plugins import Plugins
from matplotlib import use as matplotlib_use
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

matplotlib_use("Agg")

load_dotenv()

OmegaConf.register_new_resolver(
    "random",
    lambda min_value, max_value: random.randint(min_value, max_value),
)
OmegaConf.register_new_resolver(
    "version",
    lambda: generate_unused_run_id(),
    use_cache=True,
)
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(PretrainSearchpathPlugin)


def _shape_from_value(value):
    """Extract the 3D spatial shape from a tensor, array, or nested dictionary."""
    if isinstance(value, dict):
        preferred_keys = (
            "image",
            "data",
            "tensor",
            "volume",
            "input",
        )

        for key in preferred_keys:
            if key in value:
                shape = _shape_from_value(value[key])
                if shape is not None:
                    return shape

        for nested_value in value.values():
            shape = _shape_from_value(nested_value)
            if shape is not None:
                return shape

        return None

    if isinstance(value, (torch.Tensor, np.ndarray)):
        shape = tuple(int(dimension) for dimension in value.shape)

        if len(shape) == 3:
            return shape

        # Assume dimensions preceding the final three are channel/batch dimensions.
        if len(shape) > 3:
            return shape[-3:]

    return None


def _extract_voxel_shape(file_path: str):
    if file_path.endswith((".nii", ".nii.gz")):
        nii = nib.load(file_path)

        if len(nii.shape) < 3:
            raise ValueError(
                f"Expected at least 3 dimensions for {file_path}, "
                f"got {nii.shape}"
            )

        voxel_shape = tuple(int(dimension) for dimension in nii.shape[:3])

    elif file_path.endswith(".pt"):
        loaded = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )
        voxel_shape = _shape_from_value(loaded)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if voxel_shape is None:
        raise ValueError(f"Could not determine voxel shape for {file_path}")

    if len(voxel_shape) != 3:
        raise ValueError(
            f"Expected a 3D voxel shape for {file_path}, got {voxel_shape}"
        )

    return voxel_shape


def _print_axis_outliers(
    file_paths,
    voxel_shape_array: np.ndarray,
    number_of_outliers: int = 5,
) -> None:
    """
    Print the samples farthest from the median along each spatial axis.

    This identifies unusually small and unusually large dimensions.
    """
    axis_names = ("X", "Y", "Z")
    number_to_print = min(number_of_outliers, len(file_paths))

    for axis_index, axis_name in enumerate(axis_names):
        values = voxel_shape_array[:, axis_index]
        median = np.median(values)
        distances = np.abs(values - median)

        outlier_indices = np.argsort(distances)[-number_to_print:][::-1]

        print(
            f"\nTop {number_to_print} outliers for axis {axis_name} "
            f"(median={median:g}):"
        )

        for rank, sample_index in enumerate(outlier_indices, start=1):
            shape = tuple(int(value) for value in voxel_shape_array[sample_index])

            print(
                f"{rank}. path={file_paths[sample_index]}\n"
                f"   shape={shape}, "
                f"{axis_name}={values[sample_index]}, "
                f"distance_from_median={distances[sample_index]:g}"
            )


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_split = load_json(cfg.train_split_path)[cfg.data.fold]
    dataset = PretrainDataset(train_split)

    file_paths = list(dataset.files["train"])

    if not file_paths:
        raise ValueError("The training split contains no input files.")

    voxel_shapes = [
        _extract_voxel_shape(file_path)
        for file_path in file_paths
    ]
    voxel_shape_array = np.asarray(voxel_shapes, dtype=np.int64)

    _print_axis_outliers(
        file_paths=file_paths,
        voxel_shape_array=voxel_shape_array,
        number_of_outliers=5,
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        voxel_shape_array[:, 0],
        voxel_shape_array[:, 1],
        voxel_shape_array[:, 2],
        c=np.arange(len(voxel_shape_array)),
        cmap="viridis",
        s=12,
        alpha=0.8,
    )

    ax.set_xlabel("Number of voxels — X")
    ax.set_ylabel("Number of voxels — Y")
    ax.set_zlabel("Number of voxels — Z")
    ax.set_title("Pretrain input voxel dimensions")

    fig.colorbar(
        scatter,
        ax=ax,
        pad=0.1,
        label="Input index",
    )

    output_path = Path.cwd() / "pretrain_input_voxel_dimensions_3d.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
