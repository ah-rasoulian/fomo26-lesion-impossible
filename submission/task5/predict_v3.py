#!/usr/bin/env python3
"""FOMO26 Task 5: BrainDINO ViT-S PMG classification inference."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "last")


def _register_resolvers() -> None:
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
        description="FOMO26 Task 5 - Polymicrogyria Classification"
    )
    parser.add_argument(
        "--t1",
        "--t1w",
        type=Path,
        required=True,
        help="Path to the T1-weighted NIfTI image.",
    )
    parser.add_argument(
        "--t2",
        "--t2w",
        type=Path,
        default=None,
        help="Optional path to the T2-weighted NIfTI image.",
    )
    parser.add_argument(
        "--flair",
        type=Path,
        default=None,
        help="Optional path to the T2-FLAIR NIfTI image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination text file for the positive-class probability.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. The default uses CUDA when available.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> list[str]:
    # Preserve the canonical structural-modality order used by this launcher.
    named_paths = (
        ("T1", args.t1),
        ("T2", args.t2),
        ("FLAIR", args.flair),
    )
    selected: list[str] = []
    for modality_name, path in named_paths:
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"{modality_name} image was not found: {path}"
            )
        if not str(path).lower().endswith((".nii", ".nii.gz")):
            raise ValueError(
                f"{modality_name} must be a .nii or .nii.gz image: {path}"
            )
        selected.append(str(path.resolve()))

    if not selected:
        raise RuntimeError("No Task 5 input image was supplied.")
    return selected


def find_config_path(model_dir: Path) -> Path:
    possible_paths = (
        model_dir / "hydra" / "config.yaml",
        model_dir / ".hydra" / "config.yaml",
    )
    for path in possible_paths:
        if path.is_file():
            return path
    expected = "\n".join(f"  - {path}" for path in possible_paths)
    raise FileNotFoundError(
        "Could not find the final model Hydra configuration. "
        f"Expected one of:\n{expected}"
    )


def find_checkpoint_path(model_dir: Path, checkpoint_name: str) -> Path:
    supplied = Path(checkpoint_name)
    filename = supplied.name if supplied.suffix == ".ckpt" else f"{checkpoint_name}.ckpt"
    path = model_dir / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state_dict is not a mapping.")
    return state


def _load_full_finetuned_state(
    model_module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    checkpoint_model_keys = {
        key
        for key, value in state.items()
        if key.startswith("model.") and torch.is_tensor(value)
    }
    backbone_keys = sorted(
        key for key in checkpoint_model_keys if ".backbone." in key
    )
    classifier_keys = sorted(
        key for key in checkpoint_model_keys if ".classifier." in key
    )
    if not backbone_keys:
        raise RuntimeError("Checkpoint contains no backbone tensors.")
    if not classifier_keys:
        raise RuntimeError("Checkpoint contains no classifier tensors.")

    incompatible = model_module.load_state_dict(state, strict=False)
    metric_prefixes = (
        "train_metrics.",
        "val_metrics.",
        "test_metrics.",
    )
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(metric_prefixes)
    ]
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(metric_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Final checkpoint is incompatible with the bundled code. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}."
        )

    restored = model_module.state_dict()
    mismatched = []
    for key in sorted(checkpoint_model_keys):
        if key not in restored or not torch.equal(
            state[key].detach().cpu(), restored[key].detach().cpu()
        ):
            mismatched.append(key)
    if mismatched:
        raise RuntimeError(
            "Checkpoint tensors were not restored exactly: "
            f"{mismatched[:20]}"
        )

    classifier_hash = hashlib.sha256()
    for key in classifier_keys:
        classifier_hash.update(key.encode("utf-8"))
        classifier_hash.update(
            state[key]
            .detach()
            .cpu()
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

    print(
        "Checkpoint restoration verified exactly: "
        f"{len(backbone_keys)} backbone tensors and "
        f"{len(classifier_keys)} classifier tensors."
    )
    print(f"Classifier fingerprint: {classifier_hash.hexdigest()}")


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
    model = instantiate(cfg.model._cls_net, num_classes=2)
    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        weights=None,
        compile_mode=None,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        loss_weight=None,
        label_smoothing=0.0,
        positive_class_index=1,
        test_output_path=None,
        log_image_every_n_epochs=0,
    )
    _load_full_finetuned_state(
        model_module, _checkpoint_state(checkpoint_payload)
    )
    model_module.eval()
    return model_module


def _trainer(device: str) -> Trainer:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Run with --nv."
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


def predict_probability(args: argparse.Namespace) -> float:
    modality_paths = validate_inputs(args)
    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = find_checkpoint_path(MODEL_DIR, CHECKPOINT_NAME)
    cfg = OmegaConf.load(config_path)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise TypeError("The checkpoint payload is not a mapping.")

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA256: {_file_sha256(checkpoint_path)}")
    print(
        "Checkpoint training state: "
        f"epoch={payload.get('epoch')}, "
        f"global_step={payload.get('global_step')}"
    )

    model_module = _build_model_module(cfg, payload)
    data_module = instantiate(
        cfg.lightning._data_module,
        batch_size=1,
        num_workers=0,
        train_split=[],
        val_split=[],
        test_samples=[],
        predict_samples=modality_paths,
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

    # Binary predict_step already returns the positive-class probability.
    probabilities = []
    for output in outputs:
        if output is not None:
            probabilities.extend(
                torch.as_tensor(output)
                .detach()
                .float()
                .cpu()
                .reshape(-1)
                .tolist()
            )
    if len(probabilities) != 1:
        raise RuntimeError(
            "Expected exactly one PMG probability, but received "
            f"{len(probabilities)} values."
        )
    probability = float(probabilities[0])
    if not torch.isfinite(torch.tensor(probability)):
        raise RuntimeError(f"Non-finite PMG probability: {probability}")
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(f"Invalid PMG probability: {probability}")
    return probability


def main() -> int:
    _register_resolvers()
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability = predict_probability(args)

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(f"{probability:.6f}\n", encoding="utf-8")
    temporary.replace(output_path)

    print(f"Prediction saved to {output_path}")
    print(f"Polymicrogyria probability: {probability:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
