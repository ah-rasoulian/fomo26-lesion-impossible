#!/usr/bin/env python3
"""FOMO26 Task 1: BrainDINO ViT-S infarct-classification inference."""

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
    """Register resolvers that may occur in the saved Hydra configuration."""
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
        description="FOMO26 Task 1 - Infarct Classification"
    )
    parser.add_argument(
        "--flair",
        type=Path,
        required=True,
        help="Path to the T2 FLAIR image.",
    )
    parser.add_argument(
        "--adc",
        type=Path,
        required=True,
        help="Path to the ADC image.",
    )
    parser.add_argument(
        "--dwi",
        type=Path,
        required=True,
        help="Path to the DWI b1000 image.",
    )
    parser.add_argument(
        "--t2s",
        type=Path,
        default=None,
        help="Path to T2* image; mutually exclusive with --swi.",
    )
    parser.add_argument(
        "--swi",
        type=Path,
        default=None,
        help="Path to SWI image; mutually exclusive with --t2s.",
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
    """Return modalities in the exact channel order used during training."""
    required = {
        "FLAIR": args.flair,
        "ADC": args.adc,
        "DWI": args.dwi,
    }
    for modality_name, image_path in required.items():
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{modality_name} image was not found: {image_path}"
            )

    if args.t2s is not None and args.swi is not None:
        raise ValueError(
            "Both --t2s and --swi were provided. Provide exactly one "
            "susceptibility image."
        )
    if args.t2s is None and args.swi is None:
        raise ValueError(
            "A susceptibility image is required. Provide --t2s or --swi."
        )

    susceptibility = args.t2s if args.t2s is not None else args.swi
    susceptibility_name = "T2*" if args.t2s is not None else "SWI"
    assert susceptibility is not None
    if not susceptibility.is_file():
        raise FileNotFoundError(
            f"{susceptibility_name} image was not found: {susceptibility}"
        )

    paths = [args.flair, args.adc, args.dwi, susceptibility]
    for image_path in paths:
        lower = str(image_path).lower()
        if not lower.endswith((".nii", ".nii.gz")):
            raise ValueError(
                f"Task 1 expects NIfTI inputs, but received: {image_path}"
            )

    return [str(path.resolve()) for path in paths]


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
    filename = supplied.name if supplied.suffix == ".ckpt" else f"{checkpoint_name}.ckpt"
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


def _load_full_finetuned_state(
    model_module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Restore and exactly verify backbone and classifier checkpoint tensors."""
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
        key
        for key in incompatible.missing_keys
        if not key.startswith(metric_prefixes)
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(metric_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Final checkpoint is incompatible with the bundled model code. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}."
        )

    restored_state = model_module.state_dict()
    mismatched = []
    for key in sorted(checkpoint_model_keys):
        if key not in restored_state:
            mismatched.append(key)
            continue
        checkpoint_tensor = state[key].detach().cpu()
        restored_tensor = restored_state[key].detach().cpu()
        if not torch.equal(checkpoint_tensor, restored_tensor):
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    model = instantiate(
        cfg.model._cls_net,
        num_classes=2,
    )

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
        model_module,
        _checkpoint_state(checkpoint_payload),
    )
    model_module.eval()
    return model_module


def _trainer(device: str) -> Trainer:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable. "
                "Run the image with singularity run --nv."
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

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("The final checkpoint payload is not a mapping.")

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA256: {_file_sha256(checkpoint_path)}")
    print(
        "Checkpoint training state: "
        f"epoch={checkpoint_payload.get('epoch')}, "
        f"global_step={checkpoint_payload.get('global_step')}"
    )

    model_module = _build_model_module(cfg, checkpoint_payload)

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

    # The new ViTClassificationModule.predict_step() already returns the
    # positive-class probability for binary classification. Do not softmax it
    # a second time.
    probabilities = []
    for batch_output in outputs:
        if batch_output is None:
            continue
        probabilities.extend(
            torch.as_tensor(batch_output)
            .detach()
            .float()
            .cpu()
            .reshape(-1)
            .tolist()
        )

    if len(probabilities) != 1:
        raise RuntimeError(
            "Expected exactly one infarct probability, but received "
            f"{len(probabilities)} values."
        )

    probability = float(probabilities[0])
    if not torch.isfinite(torch.tensor(probability)):
        raise RuntimeError(
            f"The model returned a non-finite probability: {probability}"
        )
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"Invalid infarct probability returned: {probability}"
        )
    return probability


def main() -> int:
    _register_resolvers()
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probability = predict_probability(args)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(f"{probability:.6f}\n", encoding="utf-8")
    temporary_path.replace(output_path)

    print(f"Prediction saved to {output_path}")
    print(f"Infarct probability: {probability:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
