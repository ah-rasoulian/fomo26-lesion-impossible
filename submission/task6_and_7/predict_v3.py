#!/usr/bin/env python3
"""FOMO26 Tasks 6/7: export the final student-backbone CLS token."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from asparagus.modules.data_modules.training import ClsRegDataModule
from asparagus.modules.lightning_modules.vit_clsreg_module import (
    ViTLinearProbModule,
)
from asparagus.modules.networks.vit_task_model import ViTLinearProbModel
from asparagus.modules.transforms.presets import CPU_vit_val_test_transforms


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "last")
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the raw final CLS-token embedding from BrainDINO v3."
    )
    parser.add_argument("--input", required=True, help="Input NIfTI image")
    parser.add_argument("--output", required=True, help="Output .npy file")
    return parser.parse_args()


def find_config_path(model_dir: Path) -> Path:
    candidates = (
        model_dir / "hydra" / "config.yaml",
        model_dir / ".hydra" / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find the stored Hydra config. Checked:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def resolve_checkpoint_path(model_dir: Path, checkpoint_name: str) -> Path:
    name = checkpoint_name if checkpoint_name.endswith(".ckpt") else f"{checkpoint_name}.ckpt"
    path = model_dir / "checkpoints" / name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(input_path: Path) -> np.ndarray:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image was not found: {input_path}")

    config_path = find_config_path(MODEL_DIR)
    checkpoint_path = resolve_checkpoint_path(MODEL_DIR, CHECKPOINT_NAME)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(config_path)

    # The stored config belongs to pretraining and therefore points to
    # SubjectWisePretrainDataModule. Downstream task inference must use the
    # full-volume collate contract shared by classification and regression.
    data_module = ClsRegDataModule(
        batch_size=1,
        num_workers=0,
        train_split=None,
        val_split=None,
        predict_samples=[str(input_path)],
        predict_transforms=CPU_vit_val_test_transforms(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    pretrain_model = instantiate(cfg.model._seg_net)
    if not hasattr(pretrain_model, "backbone"):
        raise AttributeError(
            "Configured pretraining model has no .backbone; expected BrainDinoViT."
        )
    model = ViTLinearProbModel(backbone=pretrain_model.backbone)
    del pretrain_model
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    predictor = ViTLinearProbModule(
        model=model,
        weights=checkpoint,
        min_backbone_load_fraction=1.0,
    )
    predictor.eval()
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        inference_mode=True,
    )
    predictions = trainer.predict(
        model=predictor,
        datamodule=data_module,
        return_predictions=True,
    )
    if len(predictions) != 1:
        raise RuntimeError(f"Expected one prediction, received {len(predictions)}")

    embedding = predictions[0].detach().cpu().reshape(-1).numpy().astype(np.float32)
    expected_dim = int(predictor.unwrap_compiled_model().backbone.embed_dim)
    if embedding.shape != (expected_dim,):
        raise RuntimeError(
            f"Expected raw CLS embedding shape ({expected_dim},), got {embedding.shape}"
        )
    if not np.all(np.isfinite(embedding)):
        raise RuntimeError("Backbone returned non-finite CLS embedding values.")

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA-256: {sha256(checkpoint_path)}")
    print(f"CLS embedding shape: {embedding.shape}")
    return embedding


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding = predict(Path(args.input))
    with output_path.open("wb") as stream:
        np.save(stream, embedding, allow_pickle=False)
    print(f"CLS embedding saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())