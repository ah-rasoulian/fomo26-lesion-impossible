from __future__ import annotations

import json
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
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_standard_experiment
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
    "random",
    lambda minimum, maximum: random.randint(minimum, maximum),
)
OmegaConf.register_new_resolver(
    "version",
    lambda: generate_unused_run_id(),
    use_cache=True,
)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "ceil_div",
    lambda numerator, denominator: (
        int(numerator) + int(denominator) - 1
    )
    // int(denominator),
)
Plugins.instance().register(FinetuneSearchpathPlugin)


def _write_result(path: str | None, result: dict) -> None:
    if not path:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)


def _flatten_predictions(prediction_batches) -> list[float]:
    probabilities: list[float] = []
    for batch_predictions in prediction_batches:
        if batch_predictions is None:
            continue
        probabilities.extend(
            float(value)
            for value in (
                torch.as_tensor(batch_predictions)
                .detach()
                .float()
                .cpu()
                .reshape(-1)
                .tolist()
            )
        )
    return probabilities


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_cls",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    refit_full_data = bool(
        cfg.training.get("refit_full_data", False)
    )
    optimization_enabled = bool(
        cfg.optimization.get("enabled", False)
    )

    if HydraConfig.get().runtime.output_dir:
        print(f"Version: {cfg.run_id}")
        print(f"Run dir: {HydraConfig.get().run.dir}")

    logging_safe_cfg = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pretrained_weights = resolve_checkpoint(cfg)
    pl.seed_everything(int(cfg.training.seed), workers=True)

    if refit_full_data:
        # For any one fold, train + validation is the complete dataset.
        train_split = list(file_store.splits["train"]) + list(
            file_store.splits["val"]
        )
        train_split = list(dict.fromkeys(train_split))
        val_split: list = []
    else:
        train_split = list(file_store.splits["train"])
        val_split = list(file_store.splits["val"])

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
    fold_checkpoint: ModelCheckpoint | None = None
    refit_checkpoint: ModelCheckpoint | None = None

    if refit_full_data:
        # No validation data exists here. best.ckpt is therefore the final
        # CV-selected epoch, rather than a metric-selected checkpoint.
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
        # On each small fold, continuous cross-entropy is more suitable for
        # checkpoint selection than a six-pair AUROC.
        fold_checkpoint = ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            save_last=True,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best",
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        )
        callbacks.append(fold_checkpoint)

    callbacks.extend(
        [
            TQDMProgressBar(
                refresh_rate=cfg.logger.log_every_n_steps
            ),
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
    gpu_tr_transforms = instantiate(
        cfg.transforms._gpu_tr_transforms,
        ndim=3,
    )

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=train_split,
        val_split=val_split,
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=[] if refit_full_data else file_store.test,
        test_transforms=cpu_val_transforms,
        use_random_datasampler=False,
        use_weighted_sampler=cfg.training.use_weighted_sampler,
        weighted_sampler_power=cfg.training.weighted_sampler_power,
        train_num_samples=cfg.training.samples_per_epoch,
        sampler_seed=cfg.training.seed,
        log_split_details=not refit_full_data,
    )

    num_classes = int(
        file_store.dataset_json["metadata"]["n_classes"]
    )
    model = instantiate(
        cfg.model._cls_net,
        num_classes=num_classes,
        trainable_backbone_blocks=(
            cfg.model.trainable_backbone_blocks
        ),
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
        min_backbone_load_fraction=(
            cfg.model.min_backbone_load_fraction
        ),
        learning_rate=head_lr,
        backbone_learning_rate=backbone_lr,
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
        loss_weight=None,
        label_smoothing=cfg.training.label_smoothing,
        log_image_every_n_epochs=(
            cfg.logger.log_images_every_n_epoch
        ),
        test_output_path=os.path.join(
            path_store.run_dir,
            "predictions",
            cfg.test_task
            + (
                f"__{cfg.data.test_split}"
                if cfg.data.test_split
                else ""
            )
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
            0
            if refit_full_data
            else cfg.training.val_steps_per_epoch
        ),
        check_val_every_n_epoch=(
            cfg.training.check_val_every_n_epoch
        ),
        accumulate_grad_batches=(
            cfg.training.accumulate_grad_batches
        ),
        use_distributed_sampler=False,
        num_sanity_val_steps=(
            0
            if refit_full_data
            else cfg.training.num_sanity_val_steps
        ),
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=cfg.training.resume_ckpt,
    )

    if refit_full_data:
        if refit_checkpoint is None:
            raise RuntimeError(
                "Final refit checkpoint callback was not configured."
            )

        best_path = refit_checkpoint.best_model_path
        if not best_path:
            best_path = str(
                Path(path_store.ckpt_save_dir) / "best.ckpt"
            )
        if not Path(best_path).is_file():
            raise RuntimeError(
                "The full-data refit completed but best.ckpt was "
                f"not found at {best_path}."
            )

        last_path = Path(path_store.ckpt_save_dir) / "last.ckpt"
        result = {
            "mode": "refit",
            "epochs": int(cfg.training.epochs),
            "num_training_cases": len(train_split),
            "checkpoint": best_path,
            "last_checkpoint": (
                str(last_path) if last_path.is_file() else None
            ),
        }
        _write_result(
            cfg.optimization.get("result_path"),
            result,
        )
        print(json.dumps(result, indent=2))
        return

    if fold_checkpoint is None:
        raise RuntimeError(
            "Validation checkpoint callback was not configured."
        )
    if fold_checkpoint.best_model_score is None:
        raise RuntimeError(
            "No val/loss value was recorded."
        )
    if not fold_checkpoint.best_model_path:
        raise RuntimeError(
            "No minimum-validation-loss checkpoint was saved."
        )

    best_checkpoint_path = fold_checkpoint.best_model_path
    payload = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    best_epoch = int(payload["epoch"]) + 1
    best_val_loss = float(
        fold_checkpoint.best_model_score.detach().cpu()
    )

    validation_predictions = trainer.predict(
        model=model_module,
        dataloaders=data_module.val_dataloader(),
        ckpt_path=best_checkpoint_path,
        return_predictions=True,
    )
    probabilities = _flatten_predictions(validation_predictions)

    # val_dataloader uses shuffle=False, so this matches prediction order.
    labels = [
        int(data_module._read_clsreg_label(str(file)))
        for file in val_split
    ]
    if len(labels) != len(probabilities):
        raise RuntimeError(
            "Validation prediction count does not match the split: "
            f"labels={len(labels)}, "
            f"probabilities={len(probabilities)}."
        )
    if any(
        probability < 0.0 or probability > 1.0
        for probability in probabilities
    ):
        raise RuntimeError(
            "predict_step must return positive-class probabilities "
            "in [0, 1] for binary classification."
        )

    result = {
        "mode": "validation",
        "fold": int(cfg.data.fold),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "labels": labels,
        "probabilities": probabilities,
        "checkpoint": best_checkpoint_path,
        "num_training_cases": len(train_split),
        "num_validation_cases": len(val_split),
    }
    _write_result(
        cfg.optimization.get("result_path"),
        result,
    )

    if not optimization_enabled and file_store.test:
        trainer.test(
            model=model_module,
            datamodule=data_module,
            ckpt_path=best_checkpoint_path,
        )


if __name__ == "__main__":
    main()
