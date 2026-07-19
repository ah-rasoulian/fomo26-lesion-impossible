#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
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


# The Apptainer definition file will copy the trained model to /app/model.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "best")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 1 - Infarct Classification"
    )

    # Input paths for each modality
    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Path to T2 FLAIR image",
    )
    parser.add_argument(
        "--adc",
        type=str,
        required=True,
        help="Path to ADC image",
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

    # Output path for predictions
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output .txt file",
    )

    return parser.parse_args()


def validate_inputs(args):
    """Validate input paths and return them in the expected channel order."""
    required_inputs = {
        "FLAIR": args.flair,
        "ADC": args.adc,
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

    # The model expects the channels in this order:
    # FLAIR, ADC, DWI, T2*/SWI.
    return [
        args.flair,
        args.adc,
        args.dwi,
        susceptibility_path,
    ]


def find_config_path(model_dir):
    """Find the Hydra configuration stored with the trained model."""
    possible_paths = [
        model_dir / "hydra" / "config.yaml",
        model_dir / ".hydra" / "config.yaml",
    ]

    for config_path in possible_paths:
        if config_path.is_file():
            return config_path

    expected_paths = "\n".join(
        f"  - {path}" for path in possible_paths
    )

    raise FileNotFoundError(
        "Could not find the model Hydra configuration. "
        "Expected one of:\n"
        f"{expected_paths}"
    )


def predict(args):
    """
    Predict infarct probability based on the provided modalities.

    Returns:
        float: Probability of positive class (infarct presence) between 0 and 1.
    """
    data = validate_inputs(args)

    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = (
        MODEL_DIR
        / "checkpoints"
        / f"{CHECKPOINT_NAME}.ckpt"
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
        predict_transforms=CPU_clsreg_val_test_transforms_crop(
            target_size=ckpt_cfg.training.target_size
        ),
        num_workers=0,
    )

    model = instantiate(
        ckpt_cfg.model._cls_net,
        input_channels=4,
        output_channels=2,
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

    with torch.inference_mode():
        outputs = trainer.predict(
            model=model_module,
            datamodule=data_module,
            return_predictions=True,
        )

    if not outputs:
        raise RuntimeError("The model returned no predictions.")

    # trainer.predict returns one tensor per prediction batch.
    logits = torch.cat(
        [
            output.detach().cpu()
            if isinstance(output, torch.Tensor)
            else torch.as_tensor(output)
            for output in outputs
        ],
        dim=0,
    )

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError(
            "Expected classification logits with shape [N, 2], "
            f"but received {tuple(logits.shape)}."
        )

    probabilities = softmax(logits, dim=1)

    # Task 1 processes one subject, so select the first subject's
    # positive-class probability.
    probability = float(probabilities[0, 1].item())

    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"Invalid infarct probability returned: {probability}"
        )

    return probability


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory if it does not exist.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Get prediction probability.
    probability = predict(args)

    # Save probability in a text file called <subject_id>.txt.
    subject_id = Path(args.output).stem
    output_file = Path(args.output).parent / f"{subject_id}.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"{probability:.3f}")

    print(f"Prediction saved to {output_file}")
    print(f"Infarct probability: {probability:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
