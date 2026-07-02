import hydra
import nibabel as nib
import numpy as np
import random
import torch
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.hydra.plugins.searchpath_plugins import PretrainSearchpathPlugin
from asparagus.paths import get_config_path
from dotenv import load_dotenv
from hydra.core.plugins import Plugins
from matplotlib import use as matplotlib_use
from matplotlib import pyplot as plt
from numbers import Real
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from gardening_tools.functional.paths.read import load_json
from gardening_tools.functional.paths.read import load_pickle

matplotlib_use("Agg")

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("version", lambda: generate_unused_run_id(), use_cache=True)
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(PretrainSearchpathPlugin)


def _spacing_from_value(value):
    if value is None:
        return None

    if hasattr(value, "get_zooms"):
        zooms = value.get_zooms()
        if len(zooms) >= 3:
            return tuple(float(v) for v in zooms[:3])

    if isinstance(value, dict):
        for key in ("spacing", "new_spacing", "original_spacing", "target_spacing"):
            if key in value:
                spacing = _spacing_from_value(value[key])
                if spacing is not None:
                    return spacing

        for key in ("nifti_metadata", "properties", "header"):
            if key in value:
                spacing = _spacing_from_value(value[key])
                if spacing is not None:
                    return spacing

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        if all(isinstance(v, Real) for v in value[:3]):
            return tuple(float(v) for v in value[:3])

    return None


def _extract_spacing(file_path: str):
    if file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        nii = nib.load(file_path)
        spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    elif file_path.endswith(".pt"):
        pkl_path = file_path.replace(".pt", ".pkl")
        spacing = None
        if Path(pkl_path).is_file():
            spacing = _spacing_from_value(load_pickle(pkl_path))
        if spacing is None:
            loaded = torch.load(file_path, map_location="cpu")
            if isinstance(loaded, dict):
                spacing = _spacing_from_value(loaded.get("properties", loaded))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if spacing is None:
        raise ValueError(f"Could not determine spacing for {file_path}")
    if len(spacing) != 3:
        raise ValueError(f"Expected 3D spacing for {file_path}, got {spacing}")
    return spacing


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_split = load_json(cfg.train_split_path)[cfg.data.fold]
    dataset = PretrainDataset(train_split)
    spacings = [_extract_spacing(file_path) for file_path in dataset.files['train']]

    spacing_array = np.asarray(spacings, dtype=np.float32)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        spacing_array[:, 0],
        spacing_array[:, 1],
        spacing_array[:, 2],
        c=np.arange(len(spacing_array)),
        cmap="viridis",
        s=12,
        alpha=0.8,
    )
    ax.set_xlabel("Spacing X")
    ax.set_ylabel("Spacing Y")
    ax.set_zlabel("Spacing Z")
    ax.set_title("Pretrain input spacings")

    output_path = Path.cwd() / "pretrain_input_spacings_3d.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
