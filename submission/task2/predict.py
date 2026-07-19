#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 2: Meningioma Binary Segmentation
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

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 2 - Meningioma Binary Segmentation"
    )

    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Path to T2 FLAIR image",
    )
    parser.add_argument(
        "--dwi",
        type=str,
        required=True,
        help="Path to DWI b1000 image",
    )
    parser.add_argument(
        "--t2s",
        type=str,
        default=None,
        help="Path to T2* image (optional; use either T2* or SWI)",
    )
    parser.add_argument(
        "--swi",
        type=str,
        default=None,
        help="Path to SWI image (optional; use either SWI or T2*)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save segmentation NIfTI",
    )

    return parser.parse_args()


def validate_inputs(args):
    """Validate inputs and return paths in the model's channel order."""
    required_inputs = {
        "FLAIR": args.flair,
        "DWI": args.dwi,
    }

    for modality_name, image_path in required_inputs.items():
        if not Path(image_path).is_file():
            raise FileNotFoundError(
                f"{modality_name} image was not found: {image_path}"
            )

    if args.t2s is not None and args.swi is not None:
        raise ValueError(
            "Both --t2s and --swi were provided. "
            "Provide one susceptibility image only."
        )

    if args.t2s is None and args.swi is None:
        raise ValueError(
            "A susceptibility image is required. "
            "Provide either --t2s or --swi."
        )

    susceptibility_path = args.t2s if args.t2s is not None else args.swi
    susceptibility_name = "T2*" if args.t2s is not None else "SWI"

    if not Path(susceptibility_path).is_file():
        raise FileNotFoundError(
            f"{susceptibility_name} image was not found: "
            f"{susceptibility_path}"
        )

    # The model expects: FLAIR, DWI, T2*/SWI.
    return [args.flair, args.dwi, susceptibility_path]


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


def predict_segmentation(args):
    """Run segmentation inference and save the restored binary prediction."""
    data = validate_inputs(args)

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
        predict_samples=data,
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
    # Pass the filename stem, then convert to an uncompressed NIfTI when the
    # requested destination uses the ".nii" extension.
    if args.output.endswith(".nii.gz"):
        output_stem = args.output[:-7]
        convert_to_uncompressed_nifti = False
    elif args.output.endswith(".nii"):
        output_stem = args.output[:-4]
        convert_to_uncompressed_nifti = True
    else:
        raise ValueError(
            "Task 2 output must use a .nii.gz or .nii extension: "
            f"{args.output}"
        )

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
    """Main execution function."""
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predict_segmentation(args)

    print(f"Segmentation saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())