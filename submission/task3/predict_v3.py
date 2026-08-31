#!/usr/bin/env python3
"""FOMO26 Task 3: BrainDINO ViT-S brain-age inference."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "best")


def _register_resolvers() -> None:
    """Register resolvers that may occur in the saved composed Hydra config."""
    resolvers = {
        "random": lambda minimum, maximum: random.randint(minimum, maximum),
        "version": lambda: "inference",
        "eval": eval,
        "ceil_div": lambda numerator, denominator: (
            int(numerator) + int(denominator) - 1
        )
        // int(denominator),
    }

    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 3 Brain Age Prediction"
    )
    parser.add_argument(
        "--t1",
        type=Path,
        required=True,
        help="Path to the input T1-weighted NIfTI image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination text file for the predicted age in years.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. The default uses CUDA when available.",
    )
    return parser.parse_args()


def find_config_path(model_dir: Path) -> Path:
    possible_paths = (
        model_dir / "hydra" / "config.yaml",
        model_dir / ".hydra" / "config.yaml",
    )

    for config_path in possible_paths:
        if config_path.is_file():
            return config_path

    expected = "\n".join(f"  - {path}" for path in possible_paths)
    raise FileNotFoundError(
        "Could not find the final model's Hydra configuration. "
        f"Expected one of:\n{expected}"
    )


def find_checkpoint_path(model_dir: Path, checkpoint_name: str) -> Path:
    supplied = Path(checkpoint_name)

    if supplied.suffix == ".ckpt":
        filename = supplied.name
    else:
        filename = f"{checkpoint_name}.ckpt"

    checkpoint_path = model_dir / "checkpoints" / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {checkpoint_path}"
        )
    return checkpoint_path


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state_dict is not a mapping.")
    return state


def _extract_checkpoint_scalar(
    payload: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    name: str,
) -> float:
    """Read a saved normalization scalar from state or hyperparameters."""
    candidates = []
    for key, value in state.items():
        if key == name or key.endswith(f".{name}"):
            tensor = torch.as_tensor(value).detach().float().cpu().reshape(-1)
            if tensor.numel() == 1:
                candidates.append((key, float(tensor.item())))

    if candidates:
        values = {value for _, value in candidates}
        if len(values) != 1:
            raise RuntimeError(
                f"Checkpoint contains inconsistent {name} values: {candidates}"
            )
        return candidates[0][1]

    hyperparameters = payload.get("hyper_parameters", {})
    if isinstance(hyperparameters, Mapping) and name in hyperparameters:
        return float(hyperparameters[name])

    raise RuntimeError(
        f"Checkpoint does not contain the required regression {name}. "
        "Use the final checkpoint produced by the standardized-target "
        "ViTRegressionModule."
    )


def _load_full_finetuned_state(
    model_module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Restore the complete backbone, regression head, and module buffers."""
    incompatible = model_module.load_state_dict(state, strict=False)

    allowed_missing_prefixes = (
        "train_metrics.",
        "val_metrics.",
        "test_metrics.",
    )
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_missing_prefixes)
    ]

    if missing or unexpected:
        raise RuntimeError(
            "Final checkpoint is incompatible with the bundled model code. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}."
        )


def _build_predict_transforms(cfg: DictConfig):
    return instantiate(
        cfg.transforms._cpu_val_transforms,
        normalize=bool(cfg.transforms.normalize),
        center_crop_fraction=float(cfg.transforms.center_crop_fraction),
    )


def _build_model_module(
    cfg: DictConfig,
    checkpoint_payload: Mapping[str, Any],
):
    state = _checkpoint_state(checkpoint_payload)
    target_mean = _extract_checkpoint_scalar(
        checkpoint_payload, state, "target_mean"
    )
    target_std = _extract_checkpoint_scalar(
        checkpoint_payload, state, "target_std"
    )

    if not math.isfinite(target_mean):
        raise RuntimeError(f"Invalid target_mean in checkpoint: {target_mean}")
    if not math.isfinite(target_std) or target_std <= 0.0:
        raise RuntimeError(f"Invalid target_std in checkpoint: {target_std}")

    # Instantiate the regression architecture, not the old classification
    # factory. trainable_backbone_blocks affects requires_grad only and does
    # not change checkpoint tensor shapes.
    model = instantiate(
        cfg.model._reg_net,
        output_dim=1,
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        weights=None,
        target_mean=target_mean,
        target_std=target_std,
        compile_mode=None,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        test_output_path=None,
        log_image_every_n_epochs=0,
    )

    _load_full_finetuned_state(model_module, state)
    model_module.eval()
    return model_module, target_mean, target_std


def _trainer(device: str) -> Trainer:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable. "
                "Run the image with apptainer --nv."
            )
        accelerator = "gpu"
    elif device == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    return Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        inference_mode=True,
    )


def predict_age(args: argparse.Namespace) -> tuple[float, float, float]:
    t1_path = args.t1.resolve()
    if not t1_path.is_file():
        raise FileNotFoundError(f"T1 image was not found: {t1_path}")
    if not str(t1_path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Task 3 expects a .nii or .nii.gz T1 image.")

    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = find_checkpoint_path(MODEL_DIR, CHECKPOINT_NAME)
    cfg = OmegaConf.load(config_path)

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("The final checkpoint payload is not a mapping.")

    model_module, target_mean, target_std = _build_model_module(
        cfg, checkpoint_payload
    )

    data_module = instantiate(
        cfg.lightning._data_module,
        batch_size=1,
        num_workers=0,
        train_split=[],
        val_split=[],
        test_samples=[],
        predict_samples=[str(t1_path)],
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        predict_transforms=_build_predict_transforms(cfg),
        use_random_datasampler=False,
        use_weighted_sampler=False,
        train_num_samples=None,
        log_split_details=False,
        persistent_workers=False,
    )

    outputs = _trainer(args.device).predict(
        model=model_module,
        datamodule=data_module,
        return_predictions=True,
    )

    if not outputs:
        raise RuntimeError("The model returned no predictions.")

    flattened = []
    for batch_output in outputs:
        if batch_output is None:
            continue
        flattened.extend(
            torch.as_tensor(batch_output)
            .detach()
            .float()
            .cpu()
            .reshape(-1)
            .tolist()
        )

    if len(flattened) != 1:
        raise RuntimeError(
            "Expected exactly one brain-age value, but received "
            f"{len(flattened)} values."
        )

    predicted_age = float(flattened[0])
    if not np.isfinite(predicted_age):
        raise RuntimeError(
            f"The model returned a non-finite brain age: {predicted_age}"
        )

    return predicted_age, target_mean, target_std


def main() -> int:
    _register_resolvers()
    args = parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_age, target_mean, target_std = predict_age(args)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(f"{predicted_age:.2f}\n", encoding="utf-8")
    temporary_path.replace(output_path)

    print(f"Brain-age prediction saved to {output_path}")
    print(f"Predicted age: {predicted_age:.2f} years")
    print(
        "Checkpoint target normalization: "
        f"mean={target_mean:.4f}, std={target_std:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
