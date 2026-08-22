#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 5: Polymicrogyria Binary Classification
"""

import argparse
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from asparagus.modules.transforms.presets import (
    CPU_vit_val_test_transforms,
)


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "best")
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", "5"))

torch.set_float32_matmul_precision("high")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 5 Polymicrogyria Classification"
    )
    parser.add_argument(
        "--t1",
        type=str,
        required=True,
        help="Path to T1-weighted image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output .txt",
    )
    return parser.parse_args()


def validate_input(args) -> str:
    """Validate and return the T1-weighted image path."""
    t1_path = Path(args.t1)

    if not t1_path.is_file():
        raise FileNotFoundError(
            f"T1 image was not found: {t1_path}"
        )

    return str(t1_path)


def find_config_path(fold_dir: Path) -> Path:
    """Find the Hydra configuration stored for one fold."""
    possible_paths = [
        fold_dir / "hydra" / "config.yaml",
        fold_dir / ".hydra" / "config.yaml",
    ]

    for config_path in possible_paths:
        if config_path.is_file():
            return config_path

    expected_paths = "\n".join(
        f"  - {path}" for path in possible_paths
    )

    raise FileNotFoundError(
        f"Could not find the Hydra configuration for {fold_dir}.\n"
        f"Expected one of:\n{expected_paths}"
    )


def find_checkpoint_path(fold_dir: Path) -> Path:
    """Find the selected checkpoint for one fold."""
    checkpoint_path = (
        fold_dir
        / "checkpoints"
        / f"{CHECKPOINT_NAME}.ckpt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found for {fold_dir.name}: "
            f"{checkpoint_path}"
        )

    return checkpoint_path


def find_fold_directories(model_dir: Path) -> list[Path]:
    """Find and validate all cross-validation fold directories."""
    fold_dirs = [
        model_dir / f"fold_{fold_index}"
        for fold_index in range(NUM_FOLDS)
    ]

    missing = [
        str(fold_dir)
        for fold_dir in fold_dirs
        if not fold_dir.is_dir()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}" for path in missing
        )
        raise FileNotFoundError(
            "The following fold directories are missing:\n"
            f"{formatted}"
        )

    return fold_dirs


def canonical_state_key(key: str) -> str:
    """
    Normalize torch.compile state-dict keys.

    For example:
        model._orig_mod.backbone... -> model.backbone...
    """
    return ".".join(
        part
        for part in key.split(".")
        if part != "_orig_mod"
    )


def extract_checkpoint_state_dict(
    checkpoint_path: Path,
) -> Mapping[str, torch.Tensor]:
    """Read the tensor state dictionary from a Lightning checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f"Checkpoint must be a mapping: {checkpoint_path}"
        )

    state_dict = checkpoint.get("state_dict", checkpoint)

    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            "Checkpoint state_dict must be a mapping: "
            f"{checkpoint_path}"
        )

    tensor_state = {
        str(key): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }

    if not tensor_state:
        raise RuntimeError(
            f"Checkpoint contains no tensor parameters: {checkpoint_path}"
        )

    return tensor_state


def load_full_task_checkpoint(
    model_module: torch.nn.Module,
    checkpoint_path: Path,
    fold_name: str,
) -> None:
    """
    Load the complete downstream model, including the trained classifier.

    State keys are matched after removing torch.compile's `_orig_mod`
    components, allowing compiled and uncompiled checkpoints to interoperate.
    """
    source_state = extract_checkpoint_state_dict(
        checkpoint_path
    )
    target_state = model_module.state_dict()

    source_by_canonical = {}

    for source_key, tensor in source_state.items():
        canonical_key = canonical_state_key(source_key)

        if canonical_key in source_by_canonical:
            previous_key, _ = source_by_canonical[canonical_key]
            raise RuntimeError(
                f"{fold_name} checkpoint contains duplicate normalized "
                f"keys:\n  - {previous_key}\n  - {source_key}"
            )

        source_by_canonical[canonical_key] = (
            source_key,
            tensor,
        )

    mapped_state = {}
    missing_model_keys = []
    wrong_shape = []
    loaded_backbone_keys = []
    loaded_classifier_keys = []

    for target_key, target_tensor in target_state.items():
        canonical_target = canonical_state_key(target_key)
        source_entry = source_by_canonical.get(
            canonical_target
        )

        is_model_parameter = (
            canonical_target == "model"
            or canonical_target.startswith("model.")
        )

        if source_entry is None:
            if is_model_parameter:
                missing_model_keys.append(target_key)
            continue

        source_key, source_tensor = source_entry

        if source_tensor.shape != target_tensor.shape:
            if is_model_parameter:
                wrong_shape.append(
                    (
                        source_key,
                        tuple(source_tensor.shape),
                        target_key,
                        tuple(target_tensor.shape),
                    )
                )
            continue

        mapped_state[target_key] = source_tensor

        if ".backbone." in canonical_target:
            loaded_backbone_keys.append(target_key)

        if ".classifier." in canonical_target:
            loaded_classifier_keys.append(target_key)

    if missing_model_keys:
        formatted = "\n".join(
            f"  - {key}" for key in missing_model_keys
        )
        raise RuntimeError(
            f"{fold_name} is missing task-model parameters:\n"
            f"{formatted}"
        )

    if wrong_shape:
        formatted = "\n".join(
            (
                f"  - checkpoint {source_key} {source_shape} -> "
                f"model {target_key} {target_shape}"
            )
            for (
                source_key,
                source_shape,
                target_key,
                target_shape,
            ) in wrong_shape
        )
        raise RuntimeError(
            f"{fold_name} contains parameters with incompatible shapes:\n"
            f"{formatted}"
        )

    if not loaded_backbone_keys:
        raise RuntimeError(
            f"No backbone parameters were loaded for {fold_name}."
        )

    if not loaded_classifier_keys:
        raise RuntimeError(
            f"No trained classifier parameters were loaded for {fold_name}. "
            "The checkpoint may be a backbone checkpoint rather than a "
            "downstream Task 5 checkpoint."
        )

    incompatible = model_module.load_state_dict(
        mapped_state,
        strict=False,
    )

    missing_after_load = [
        key
        for key in incompatible.missing_keys
        if canonical_state_key(key).startswith("model.")
    ]

    if missing_after_load:
        formatted = "\n".join(
            f"  - {key}" for key in missing_after_load
        )
        raise RuntimeError(
            f"{fold_name} task-model parameters remained missing after "
            f"loading:\n{formatted}"
        )


def concatenate_prediction_batches(
    outputs: Sequence,
) -> torch.Tensor:
    """Concatenate Lightning prediction batches."""
    if not outputs:
        raise RuntimeError("The model returned no predictions.")

    tensors = []

    for output in outputs:
        if isinstance(output, dict):
            if "probabilities" in output:
                output = output["probabilities"]
            elif "probability" in output:
                output = output["probability"]
            elif "logits" in output:
                output = output["logits"]
            else:
                raise RuntimeError(
                    "Prediction output dictionary does not contain "
                    "'probabilities', 'probability', or 'logits'."
                )

        if isinstance(output, torch.Tensor):
            tensor = output.detach().float().cpu()
        else:
            tensor = torch.as_tensor(
                output,
                dtype=torch.float32,
            )

        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)

        tensors.append(tensor)

    predictions = torch.cat(tensors, dim=0)

    if predictions.ndim != 2:
        raise RuntimeError(
            "Expected predictions with shape [N, 1] or [N, 2], "
            f"but received {tuple(predictions.shape)}."
        )

    if predictions.shape[1] not in (1, 2):
        raise RuntimeError(
            "Expected one positive-class probability or two logits per "
            f"subject, but received {tuple(predictions.shape)}."
        )

    return predictions


def build_data_module(
    ckpt_cfg,
    t1_path: str,
):
    """Build the single-subject Task 5 prediction data module."""
    return instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        predict_samples=[t1_path],
        predict_transforms=CPU_vit_val_test_transforms(),
        num_workers=0,
    )


def predict_one_fold(
    fold_dir: Path,
    t1_path: str,
    trainer: Trainer,
    output_path: str,
) -> float:
    """Predict the Task 5 probability using one fold."""
    config_path = find_config_path(fold_dir)
    checkpoint_path = find_checkpoint_path(fold_dir)
    ckpt_cfg = OmegaConf.load(config_path)

    data_module = build_data_module(
        ckpt_cfg=ckpt_cfg,
        t1_path=t1_path,
    )

    # The new ViT factory obtains num_classes and all backbone settings
    # from the saved Hydra configuration.
    model = instantiate(
        ckpt_cfg.model._cls_net,
    )

    # Do not pass the downstream checkpoint through `weights`, because that
    # argument invokes pretrained-backbone loading and can leave the
    # classification head randomly initialized.
    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=None,
        test_output_path=output_path,
    )

    load_full_task_checkpoint(
        model_module=model_module,
        checkpoint_path=checkpoint_path,
        fold_name=fold_dir.name,
    )

    model_module.eval()

    with torch.inference_mode():
        outputs = trainer.predict(
            model=model_module,
            datamodule=data_module,
            return_predictions=True,
        )

    predictions = concatenate_prediction_batches(outputs)

    if predictions.shape[0] != 1:
        raise RuntimeError(
            f"{fold_dir.name} returned predictions for "
            f"{predictions.shape[0]} subjects; expected exactly one."
        )

    if predictions.shape[1] == 1:
        # The Lightning predict_step already returned the positive-class
        # probability.
        probability = float(
            predictions[0, 0].item()
        )

        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                f"{fold_dir.name} returned one value outside [0, 1]. "
                "A single prediction must be the positive-class "
                f"probability, but received {probability}."
            )
    else:
        # Compatibility fallback if predict_step returns raw two-class logits.
        probability = float(
            torch.softmax(
                predictions,
                dim=1,
            )[0, 1].item()
        )

    if not np.isfinite(probability):
        raise RuntimeError(
            f"{fold_dir.name} returned a non-finite probability: "
            f"{probability}"
        )

    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"Invalid probability returned by {fold_dir.name}: "
            f"{probability}"
        )

    return probability


def predict(args) -> float:
    """
    Predict the ensemble polymicrogyria probability.

    Returns:
        Mean positive-class probability across all folds.
    """
    t1_path = validate_input(args)
    fold_dirs = find_fold_directories(MODEL_DIR)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    fold_probabilities = []

    for fold_dir in fold_dirs:
        probability = predict_one_fold(
            fold_dir=fold_dir,
            t1_path=t1_path,
            trainer=trainer,
            output_path=args.output,
        )
        fold_probabilities.append(probability)

    ensemble_probability = float(
        np.mean(fold_probabilities)
    )

    if not np.isfinite(ensemble_probability):
        raise RuntimeError(
            "The ensemble returned a non-finite probability."
        )

    if not 0.0 <= ensemble_probability <= 1.0:
        raise RuntimeError(
            "Invalid ensemble polymicrogyria probability returned: "
            f"{ensemble_probability}"
        )

    return ensemble_probability


def main():
    """Run Task 5 inference and save the positive-class probability."""
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability = predict(args)

    # Preserve the original challenge classification-output format exactly.
    output_file = (
        output_path.parent
        / f"{output_path.stem}.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"{probability:.3f}")

    print(f"Prediction saved to {output_file}")
    print(f"Polymicrogyria probability: {probability:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
