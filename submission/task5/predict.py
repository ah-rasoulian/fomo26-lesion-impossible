#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 5: Polymicrogyria Binary Classification
"""

import argparse
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch.nn.functional import softmax

from asparagus.modules.transforms.presets import (
    CPU_clsreg_val_test_transforms_crop,
)
from asparagus.pipeline.auto_configuration.checkpoint import (
    load_checkpoint_state_dict,
)


# The Apptainer definition file copies the trained model to /app/model.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "best")

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2


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


def find_config_path(model_dir):
    """Find the Hydra configuration stored with the trained model."""
    possible_paths = [
        model_dir / "hydra" / "config.yaml",
        model_dir / ".hydra" / "config.yaml",
    ]

    for config_path in possible_paths:
        if config_path.is_file():
            return config_path

    expected_paths = "\n".join(f"  - {path}" for path in possible_paths)
    raise FileNotFoundError(
        "Could not find the model Hydra configuration. "
        "Expected one of:\n"
        f"{expected_paths}"
    )


def predict(args):
    """Predict the positive-class polymicrogyria probability from T1w."""
    t1_path = Path(args.t1)
    if not t1_path.is_file():
        raise FileNotFoundError(f"T1 image was not found: {t1_path}")

    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = (
        MODEL_DIR / "checkpoints" / f"{CHECKPOINT_NAME}.ckpt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {checkpoint_path}"
        )

    ckpt_cfg = OmegaConf.load(config_path)

    data_module = instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        predict_samples=[str(t1_path)],
        predict_transforms=CPU_clsreg_val_test_transforms_crop(
            target_size=ckpt_cfg.training.target_size
        ),
        num_workers=0,
    )

    model = instantiate(
        ckpt_cfg.model._cls_net,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(checkpoint_path),
        test_output_path=args.output,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    outputs = trainer.predict(
        model=model_module,
        datamodule=data_module,
        return_predictions=True,
    )

    if not outputs:
        raise RuntimeError("The model returned no predictions.")

    logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)

    if logits.ndim != 2 or logits.shape != (1, OUTPUT_CHANNELS):
        raise RuntimeError(
            "Expected classification logits with shape [1, 2], "
            f"but received {tuple(logits.shape)}."
        )

    probability = float(softmax(logits, dim=1)[0, 1].item())
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            "The model returned an invalid positive-class probability: "
            f"{probability}"
        )

    return probability


def main():
    """Run Task 5 inference and save the positive-class probability."""
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probability = predict(args)

    # Match the challenge classification-output convention.
    output_file = output_path.parent / f"{output_path.stem}.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"{probability:.3f}")

    print(f"Prediction saved to {output_file}")
    print(f"Polymicrogyria probability: {probability:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())