#!/usr/bin/env python3
"""FOMO26 Task 2 v3: full-volume BrainDINO ViT segmentation inference."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import Compose

from asparagus.functional.loading import MODALITY_TO_ID, get_modality_id
from asparagus.modules.datasets.TrainDataset import (
    SingleSubjectPredictDataset,
    full_volume_task_collate,
)


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "best")
INFERENCE_PRECISION = os.environ.get("INFERENCE_PRECISION", "auto").lower()

OUTPUT_CHANNELS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FOMO26 Task 2 - full-volume BrainDINO ViT meningioma "
            "segmentation"
        )
    )
    parser.add_argument("--flair", required=True, help="T2 FLAIR NIfTI")
    parser.add_argument("--dwi", required=True, help="DWI b1000 NIfTI")
    parser.add_argument(
        "--t2s",
        default=None,
        help="T2-star NIfTI (provide either --t2s or --swi)",
    )
    parser.add_argument(
        "--swi",
        default=None,
        help="SWI NIfTI (provide either --swi or --t2s)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination binary segmentation (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Foreground probability threshold (default: 0.5)",
    )
    return parser.parse_args()


def _require_nifti(path: str, name: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{name} image was not found: {result}")
    if not str(result).endswith((".nii", ".nii.gz")):
        raise ValueError(f"{name} must be a .nii or .nii.gz file: {result}")
    return result


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], nib.spatialimages.SpatialImage]:
    if (args.t2s is None) == (args.swi is None):
        raise ValueError("Provide exactly one of --t2s and --swi.")
    if not 0.0 < args.foreground_threshold < 1.0:
        raise ValueError("--foreground-threshold must be in (0, 1).")
    if not args.output.endswith((".nii", ".nii.gz")):
        raise ValueError("--output must end in .nii or .nii.gz.")

    flair = _require_nifti(args.flair, "FLAIR")
    dwi = _require_nifti(args.dwi, "DWI")
    susceptibility = _require_nifti(
        args.t2s if args.t2s is not None else args.swi,
        "T2*" if args.t2s is not None else "SWI",
    )

    paths = [str(flair), str(dwi), str(susceptibility)]
    modality_names = [
        "FLAIR",
        "DWI",
        "T2S" if args.t2s is not None else "SWI",
    ]

    # The dataset also validates the grid. Doing it here produces a concise
    # challenge-facing error before the model is initialized.
    reference = nib.load(str(flair))
    reference_shape = tuple(int(value) for value in reference.shape[:3])
    reference_affine = np.asarray(reference.affine)
    for name, path in zip(modality_names[1:], paths[1:]):
        image = nib.load(path)
        shape = tuple(int(value) for value in image.shape[:3])
        if shape != reference_shape:
            raise ValueError(
                f"{name} shape {shape} does not match FLAIR "
                f"shape {reference_shape}. Inputs must already be registered."
            )
        if not np.allclose(image.affine, reference_affine, atol=1e-4, rtol=1e-4):
            raise ValueError(
                f"{name} affine does not match the FLAIR affine. "
                "Inputs must be on the same registered voxel grid."
            )
    return paths, modality_names, reference


def find_config_path(model_dir: Path) -> Path:
    candidates = (
        model_dir / "hydra" / "config.yaml",
        model_dir / ".hydra" / "config.yaml",
        model_dir / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    locations = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the Hydra config stored with the model. Checked:\n"
        f"{locations}"
    )


def find_checkpoint_path(model_dir: Path, checkpoint_name: str) -> Path:
    requested = Path(checkpoint_name)
    names = [requested.name]
    if requested.suffix != ".ckpt":
        names.append(f"{requested.name}.ckpt")

    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.extend(Path(name) for name in names)
    else:
        for name in names:
            candidates.extend(
                (model_dir / "checkpoints" / name, model_dir / name)
            )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    locations = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find the checkpoint. Checked:\n{locations}")


def _config_value(
    cfg: DictConfig,
    path: str,
    default: Any,
) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def _normalized_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _modality_id(
    path: str,
    display_name: str,
    aliases: Sequence[str],
) -> int:
    """Resolve a modality ID from the explicit CLI role, not only filename."""
    normalized_mapping = {
        _normalized_name(str(key)): int(value)
        for key, value in MODALITY_TO_ID.items()
    }
    for alias in aliases:
        key = _normalized_name(alias)
        if key in normalized_mapping:
            return normalized_mapping[key]

    unknown = int(MODALITY_TO_ID.get("UNKNOWN", 0))
    inferred = int(torch.as_tensor(get_modality_id([path])).reshape(-1)[0])
    if inferred != unknown:
        return inferred

    available = ", ".join(sorted(str(key) for key in MODALITY_TO_ID))
    raise RuntimeError(
        f"Could not resolve the model modality ID for {display_name}. "
        f"Available modality keys are: {available}"
    )


def modality_ids(paths: Sequence[str], names: Sequence[str]) -> torch.Tensor:
    aliases = {
        "FLAIR": ("FLAIR", "T2FLAIR", "T2_FLAIR"),
        "DWI": ("DWI", "DWIB1000", "DWI_B1000", "B1000"),
        "T2S": ("T2S", "T2STAR", "T2_STAR", "T2*"),
        "SWI": ("SWI",),
    }
    return torch.tensor(
        [
            _modality_id(path, name, aliases[name])
            for path, name in zip(paths, names)
        ],
        dtype=torch.long,
    )


def build_batch(
    paths: Sequence[str],
    names: Sequence[str],
    normalize: bool,
) -> dict[str, Any]:
    # CPU validation used Torch_Normalize followed by label validation. At
    # prediction time there is no label, so apply exactly the normalization
    # component and let the prediction dataset retain native geometry.
    dataset = SingleSubjectPredictDataset(
        list(paths),
        transforms=Compose([Torch_Normalize(normalize=normalize)]),
    )
    sample = dataset[0]
    sample["info"]["modality"] = modality_ids(paths, names)
    sample["info"]["channel_mask"] = torch.ones(
        len(paths), dtype=torch.bool
    )
    return full_volume_task_collate([sample])


def _strip_compile_prefix(key: str) -> str:
    # torch.compile can insert _orig_mod into model paths. v3 inference is
    # deliberately eager, so normalize those keys before strict loading.
    return key.replace("._orig_mod.", ".").removeprefix("_orig_mod.")


def load_full_lightning_checkpoint(
    model_module: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping.")

    normalized = {
        _strip_compile_prefix(str(key)): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    try:
        model_module.load_state_dict(normalized, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The complete v3 checkpoint could not be loaded strictly. "
            "Make sure /app/asparagus, the saved Hydra config, and best.ckpt "
            "all come from the same final-feature segmentation run."
        ) from error


def build_model_module(cfg: DictConfig, checkpoint_path: Path):
    num_classes = int(_config_value(cfg, "model.output_channels", OUTPUT_CHANNELS))
    if num_classes != OUTPUT_CHANNELS:
        raise ValueError(
            f"Task 2 requires {OUTPUT_CHANNELS} output channels, but the "
            f"saved config requests {num_classes}."
        )

    model = instantiate(
        cfg.model._seg_net,
        output_channels=num_classes,
        # Training-time freezing only controls gradients. Freeze everything
        # for inference without changing any forward computation or weights.
        freeze_backbone=True,
        trainable_backbone_blocks=0,
    )
    module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        weights=None,
        compile_mode=None,
        num_classes=num_classes,
        label_key=str(_config_value(cfg, "training.label_key", "SEG_label")),
        ce_weight=float(_config_value(cfg, "training.ce_weight", 0.35)),
        dice_weight=float(_config_value(cfg, "training.dice_weight", 0.65)),
        class_weights=_config_value(cfg, "training.class_weights", None),
        include_background_in_dice=bool(
            _config_value(cfg, "training.include_background_in_dice", False)
        ),
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        log_image_every_n_epochs=0,
    )
    load_full_lightning_checkpoint(module, checkpoint_path)
    module.eval()
    return module


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _autocast_settings(device: torch.device) -> tuple[bool, torch.dtype]:
    if INFERENCE_PRECISION not in {"auto", "fp32", "bf16"}:
        raise ValueError(
            "INFERENCE_PRECISION must be one of: auto, fp32, bf16."
        )
    use_bf16 = device.type == "cuda" and INFERENCE_PRECISION != "fp32"
    if INFERENCE_PRECISION == "bf16" and device.type != "cuda":
        raise RuntimeError("INFERENCE_PRECISION=bf16 requires a CUDA GPU.")
    return use_bf16, torch.bfloat16


def save_binary_prediction(
    probabilities: torch.Tensor,
    reference: nib.spatialimages.SpatialImage,
    output_path: str,
    threshold: float,
) -> None:
    if probabilities.ndim != 5 or probabilities.shape[:2] != (1, 2):
        raise RuntimeError(
            "Expected probabilities [1,2,H,W,D], got "
            f"{tuple(probabilities.shape)}."
        )
    foreground = probabilities[0, 1]
    prediction = (foreground >= threshold).to(torch.uint8).cpu().numpy()
    reference_shape = tuple(int(value) for value in reference.shape[:3])
    if prediction.shape != reference_shape:
        raise RuntimeError(
            f"Prediction shape {prediction.shape} does not match the "
            f"FLAIR shape {reference_shape}."
        )

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_data_shape(reference_shape)
    output_image = nib.Nifti1Image(
        prediction,
        np.asarray(reference.affine),
        header=header,
    )
    output_image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output_image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output_image, output_path)


def predict_segmentation(args: argparse.Namespace) -> None:
    paths, names, reference = validate_inputs(args)
    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = find_checkpoint_path(MODEL_DIR, CHECKPOINT_NAME)
    cfg = OmegaConf.load(config_path)

    normalize = bool(_config_value(cfg, "transforms.normalize", True))
    batch = build_batch(paths, names, normalize=normalize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    module = build_model_module(cfg, checkpoint_path).to(device)
    batch = _move_to_device(batch, device)
    use_autocast, autocast_dtype = _autocast_settings(device)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_autocast,
        ):
            probabilities = module.predict_step(batch, batch_idx=0)

    save_binary_prediction(
        probabilities.detach().float().cpu(),
        reference=reference,
        output_path=args.output,
        threshold=float(args.foreground_threshold),
    )


def main() -> int:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predict_segmentation(args)
    print(f"Segmentation saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
