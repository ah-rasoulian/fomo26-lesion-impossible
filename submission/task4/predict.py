#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 4: Trigeminal Neuralgia Multiclass Segmentation
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from asparagus.modules.transforms.presets import CPU_seg_test_transforms
from asparagus.pipeline.auto_configuration.checkpoint import (
    load_checkpoint_state_dict,
)
from gardening_tools.functional.paths.write import save_prediction_from_logits


# The Apptainer definition file copies the trained model to /app/model.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "best")

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 3


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 4 Trigeminal Multiclass Segmentation"
    )
    parser.add_argument(
        "--t2",
        type=str,
        required=True,
        help="Path to T2-weighted image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save segmentation NIfTI",
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


def get_output_stem(output_path):
    """Return the writer stem and whether `.nii` conversion is required."""
    output_path = str(output_path)

    if output_path.endswith(".nii.gz"):
        return output_path[:-7], False
    if output_path.endswith(".nii"):
        return output_path[:-4], True

    raise ValueError(
        "Task 4 output must use a .nii.gz or .nii extension: "
        f"{output_path}"
    )


def predict_segmentation(args):
    """Generate a three-class segmentation from a T2-weighted image."""
    t2_path = Path(args.t2)
    if not t2_path.is_file():
        raise FileNotFoundError(f"T2 image was not found: {t2_path}")

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
        predict_samples=[str(t2_path)],
        predict_transforms=CPU_seg_test_transforms(
            patch_size=ckpt_cfg.training.patch_size
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
        inference_patch_size=ckpt_cfg.training.patch_size,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    logits, properties = trainer.predict(
        model=model_module,
        datamodule=data_module,
    )[0]

    # save_prediction_from_logits always appends ".nii.gz" internally.
    output_stem, convert_to_uncompressed_nifti = get_output_stem(args.output)

    save_prediction_from_logits(
        logits.numpy(),
        output_stem,
        properties=properties,
    )

    if convert_to_uncompressed_nifti:
        temporary_output = Path(f"{output_stem}.nii.gz")
        output_image = nib.load(temporary_output)
        nib.save(output_image, args.output)
        temporary_output.unlink()


def main():
    """Run Task 4 inference and save the multiclass segmentation."""
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predict_segmentation(args)

    print(f"Segmentation saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())