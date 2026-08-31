from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import hydra
import lightning as pl
import torch
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import FinetuneSearchpathPlugin
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import resolve_checkpoint
from asparagus.pipeline.auto_configuration.experiment_setup import (
    prepare_standard_experiment,
)
from asparagus.pipeline.auto_configuration.logging import logging
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf


load_dotenv()

OmegaConf.register_new_resolver(
    "random", lambda minimum, maximum: random.randint(minimum, maximum)
)
OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True
)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "ceil_div",
    lambda numerator, denominator: (int(numerator) + int(denominator) - 1)
    // int(denominator),
)
Plugins.instance().register(FinetuneSearchpathPlugin)


def _calculate_target_statistics(
    files: list,
) -> tuple[float, float]:
    targets = []

    for file in files:
        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(data, (tuple, list)) or len(data) < 2:
            raise RuntimeError(
                f"Expected (image, target) in {file}."
            )

        target = torch.as_tensor(
            data[1],
            dtype=torch.float32,
        ).reshape(-1)

        if target.numel() != 1:
            raise RuntimeError(
                "Brain-age regression expects one target per subject, "
                f"but {file} contains {target.numel()} values."
            )

        value = float(target.item())

        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite regression target in {file}: {value}."
            )

        targets.append(value)

    if len(targets) < 2:
        raise RuntimeError(
            "At least two training subjects are required to "
            "calculate target normalization."
        )

    tensor = torch.tensor(targets, dtype=torch.float32)
    mean = float(tensor.mean())
    std = float(tensor.std(unbiased=False))

    if std <= 0.0 or not math.isfinite(std):
        raise RuntimeError(
            f"Training target standard deviation is invalid: {std}."
        )

    return mean, std

def _write_result(path: str | None, result: dict) -> None:
    """Atomically write an optional optimization/result JSON file."""
    if not path:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)


def _flatten_regression_predictions(prediction_batches: list) -> list[float]:
    """Convert scalar-regression prediction batches into a flat float list."""
    predictions: list[float] = []

    for batch in prediction_batches:
        if isinstance(batch, dict):
            for key in ("predictions", "prediction", "output", "logits"):
                if key in batch:
                    batch = batch[key]
                    break
            else:
                raise RuntimeError(
                    "Regression predict_step returned a dictionary without a "
                    "recognized prediction key."
                )

        tensor = torch.as_tensor(batch).detach().float().cpu()

        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        elif tensor.ndim != 1:
            raise RuntimeError(
                "finetune_reg.py expects scalar regression predictions with "
                f"shape [B] or [B, 1], but received {tuple(tensor.shape)}."
            )

        predictions.extend(float(value) for value in tensor.tolist())

    return predictions


def _read_scalar_targets(data_module, samples: list) -> list[float]:
    """Read scalar regression targets in the same order as the val loader."""
    targets: list[float] = []

    for sample in samples:
        target = data_module._read_clsreg_label(sample)
        tensor = torch.as_tensor(target).detach().float().cpu().reshape(-1)
        if tensor.numel() != 1:
            raise RuntimeError(
                "finetune_reg.py expects one scalar target per subject, but "
                f"received {tensor.numel()} values for sample {sample!r}."
            )
        targets.append(float(tensor.item()))

    return targets


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_reg",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    refit_full_data = bool(cfg.training.get("refit_full_data", False))
    optimization_enabled = bool(cfg.optimization.get("enabled", False))

    if HydraConfig.get().runtime.output_dir:
        print(f"Version: {cfg.run_id}")
        print(f"Run dir: {HydraConfig.get().run.dir}")

    logging_safe_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pretrained_weights = resolve_checkpoint(cfg)
    pl.seed_everything(int(cfg.training.seed), workers=True)

    if refit_full_data:
        train_split = list(file_store.splits["train"]) + list(
            file_store.splits["val"]
        )
        # Preserve order while guarding against duplicated subjects.
        train_split = list(dict.fromkeys(train_split))
        val_split = []
    else:
        train_split = list(file_store.splits["train"])
        val_split = list(file_store.splits["val"])

    target_mean, target_std = _calculate_target_statistics(
        train_split
    )

    print(
        f"Training age normalization: "
        f"mean={target_mean:.4f}, std={target_std:.4f}"
    )

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project=cfg.logger.wandb_project,
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_to_stdout=cfg.logger.log_to_stdout,
    )

    callbacks = []
    best_checkpoint: ModelCheckpoint | None = None
    refit_checkpoint: ModelCheckpoint | None = None

    if refit_full_data:
        # There is no held-out metric during the final full-data fit. Save the
        # final epoch as both best.ckpt and last.ckpt.
        refit_checkpoint = ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            monitor=None,
            save_top_k=1,
            filename="best",
            every_n_epochs=1,
            save_last=True,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
        callbacks.append(refit_checkpoint)
    else:
        # For regression, validation loss is the stable checkpoint-selection
        # metric. The Lightning regression module normally logs val/loss.
        best_checkpoint = ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best",
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        )
        callbacks.append(best_checkpoint)

    callbacks.extend(
        [
            TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps),
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=True,
                log_weight_decay=True,
            ),
        ]
    )

    cpu_tr_transforms = instantiate(
        cfg.transforms._cpu_tr_transforms,
        normalize=cfg.transforms.normalize,
        center_crop_fraction=cfg.transforms.center_crop_fraction,
    )
    cpu_val_transforms = instantiate(
        cfg.transforms._cpu_val_transforms,
        normalize=cfg.transforms.normalize,
        center_crop_fraction=cfg.transforms.center_crop_fraction,
    )
    gpu_tr_transforms = instantiate(cfg.transforms._gpu_tr_transforms, ndim=3)

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=train_split,
        val_split=val_split,
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=[] if refit_full_data else file_store.test,
        test_transforms=cpu_val_transforms,
        # A class-weighted sampler is inappropriate for continuous targets.
        # Random sampling with replacement still permits a configured number
        # of augmented samples per epoch on a small regression dataset.
        use_random_datasampler=True,
        use_weighted_sampler=False,
        train_num_samples=cfg.training.samples_per_epoch,
        sampler_seed=cfg.training.seed,
        log_split_details=not refit_full_data,
    )

    output_dim = int(cfg.model.get("output_dim", 1))
    if output_dim != 1:
        raise ValueError(
            "This launcher currently supports scalar regression only. Set "
            "model.output_dim: 1 for brain-age regression."
        )

    model = instantiate(
        cfg.model._reg_net,
        output_dim=output_dim,
        trainable_backbone_blocks=cfg.model.trainable_backbone_blocks,
    )

    head_lr = float(cfg.model.finetune_lr)
    backbone_lr = (
        head_lr * float(cfg.model.backbone_lr_multiplier)
        if int(cfg.model.trainable_backbone_blocks) > 0
        else 0.0
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        weights=pretrained_weights,
        min_backbone_load_fraction=cfg.model.min_backbone_load_fraction,
        learning_rate=head_lr,
        backbone_learning_rate=backbone_lr,

        target_mean=target_mean,
        target_std=target_std,

        warmup_ratio=cfg.model.warmup_ratio,
        cosine_period_ratio=cfg.model.cosine_period_ratio,
        minimum_lr=cfg.model.minimum_lr,
        optimizer=cfg.model.finetune_optim,
        weight_decay=cfg.model.finetune_weight_decay,
        momentum=cfg.model.momentum,
        nesterov=cfg.model.nesterov,
        compile_mode=cfg.model.get("compile_mode"),
        train_transforms=gpu_tr_transforms,
        val_transforms=None,
        test_transforms=None,
        log_image_every_n_epochs=cfg.logger.log_images_every_n_epoch,
        test_output_path=os.path.join(
            path_store.run_dir,
            "predictions",
            cfg.test_task
            + (f"__{cfg.data.test_split}" if cfg.data.test_split else "")
            + "__best.json",
        ),
    )

    trainer = instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        profiler=None,
        default_root_dir=path_store.run_dir,
        max_epochs=cfg.training.epochs,
        limit_train_batches=cfg.training.steps_per_epoch,
        limit_val_batches=(
            0 if refit_full_data else cfg.training.val_steps_per_epoch
        ),
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        use_distributed_sampler=False,
        num_sanity_val_steps=(
            0 if refit_full_data else cfg.training.num_sanity_val_steps
        ),
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=cfg.training.resume_ckpt,
    )

    if refit_full_data:
        if refit_checkpoint is None:
            raise RuntimeError("The full-data checkpoint callback was not created.")

        checkpoint_path = refit_checkpoint.best_model_path
        if not checkpoint_path:
            checkpoint_path = str(
                Path(path_store.ckpt_save_dir) / "last.ckpt"
            )

        if not Path(checkpoint_path).exists():
            raise RuntimeError(
                f"The final regression checkpoint was not saved: {checkpoint_path}"
            )

        _write_result(
            cfg.optimization.get("result_path"),
            {
                "mode": "refit",
                "epochs": int(cfg.training.epochs),
                "num_training_cases": len(train_split),
                "checkpoint": checkpoint_path,
                "last_checkpoint": str(
                    Path(path_store.ckpt_save_dir) / "last.ckpt"
                ),
            },
        )
        return

    if best_checkpoint is None or best_checkpoint.best_model_score is None:
        raise RuntimeError("No scalar val/loss was recorded.")
    if not best_checkpoint.best_model_path:
        raise RuntimeError("No best validation-loss checkpoint was saved.")

    payload = torch.load(
        best_checkpoint.best_model_path,
        map_location="cpu",
        weights_only=False,
    )

    prediction_batches = trainer.predict(
        model=model_module,
        dataloaders=data_module.val_dataloader(),
        ckpt_path=best_checkpoint.best_model_path,
        return_predictions=True,
    )
    if prediction_batches is None:
        raise RuntimeError("trainer.predict returned no regression predictions.")

    predictions = _flatten_regression_predictions(prediction_batches)
    targets = _read_scalar_targets(data_module, val_split)

    if len(predictions) != len(targets):
        raise RuntimeError(
            "Validation prediction count does not match target count: "
            f"{len(predictions)} predictions versus {len(targets)} targets."
        )

    absolute_errors = [
        abs(prediction - target)
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    squared_errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    val_mae = sum(absolute_errors) / len(absolute_errors)
    val_rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    result = {
        "mode": "validation",
        "fold": int(cfg.data.fold),
        "best_val_loss": float(
            best_checkpoint.best_model_score.detach().cpu()
        ),
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "best_epoch": int(payload["epoch"]) + 1,
        "checkpoint": best_checkpoint.best_model_path,
        "num_training_cases": len(train_split),
        "num_validation_cases": len(val_split),
        "targets": targets,
        "predictions": predictions,
    }
    _write_result(cfg.optimization.get("result_path"), result)

    if not optimization_enabled and file_store.test:
        trainer.test(
            model=model_module,
            datamodule=data_module,
            ckpt_path=best_checkpoint.best_model_path,
        )


if __name__ == "__main__":
    main()
