#!/usr/bin/env python3
"""
FOMO26 Challenge - Tasks 6 and 7:
Linear Probing and Bias & Fairness on Frozen Pretrained Embeddings
"""

import argparse
import os
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from asparagus.modules.transforms.presets import (
    CPU_clsreg_val_test_transforms_crop,
)
from asparagus.pipeline.auto_configuration.checkpoint import (
    load_checkpoint_state_dict,
)


# The Apptainer definition file copies the pretrained model to /app/model.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "last")

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

def parse_args():
    """Parse the shared Task 6/7 command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "FOMO26 Tasks 6 and 7 - frozen pretrained embeddings. "
            "The input volume is center-cropped before encoding."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input NIfTI",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save embeddings .npy",
    )
    return parser.parse_args()


def find_config_path(model_dir):
    """Find the Hydra configuration stored with the pretrained model."""
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
    """Compute a frozen 768-element embedding for an input MR volume."""
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image was not found: {input_path}")

    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = (
        MODEL_DIR / "checkpoints" / f"{CHECKPOINT_NAME}.ckpt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {checkpoint_path}"
        )

    # This resolver is used by the stored pretraining configuration.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    ckpt_cfg = OmegaConf.load(config_path)

    data_module = instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        predict_samples=[str(input_path)],
        predict_transforms=CPU_clsreg_val_test_transforms_crop(
            target_size=ckpt_cfg.training.patch_size
        ),
        num_workers=0,
    )

    model = instantiate(
        ckpt_cfg.model._seg_net,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(checkpoint_path),
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    embeddings = trainer.predict(
        model=model_module,
        datamodule=data_module,
        return_predictions=True,
    )[0]

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().reshape(-1).numpy()
    else:
        embeddings = np.asarray(embeddings).reshape(-1)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("The model returned non-finite embedding values.")

    return embeddings


def main():
    """Generate and save embeddings for Task 6 or Task 7."""
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = predict(args)

    # Passing a file object prevents np.save from altering the requested path.
    with open(output_path, "wb") as file:
        np.save(file, embeddings)

    print(f"Embeddings saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
