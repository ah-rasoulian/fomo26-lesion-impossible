import os
import random

import hydra
import lightning as pl
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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf


load_dotenv()

OmegaConf.register_new_resolver("random", lambda minimum, maximum: random.randint(minimum, maximum))
OmegaConf.register_new_resolver("version", lambda: generate_unused_run_id(), use_cache=True)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "ceil_div",
    lambda numerator, denominator: (int(numerator) + int(denominator) - 1)
    // int(denominator),
)
Plugins.instance().register(FinetuneSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_cls",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    if HydraConfig.get().runtime.output_dir:
        print(f"Version: {cfg.run_id}")
        print(f"Run dir: {HydraConfig.get().run.dir}")

    logging_safe_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pretrained_weights = resolve_checkpoint(cfg)
    pl.seed_everything(cfg.training.seed, workers=True)

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

    callbacks = [
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best",
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        ),
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            filename="epoch-{epoch:03d}",
            every_n_epochs=cfg.model.ckpt_every_n_epoch,
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        ),
        TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps),
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
            log_weight_decay=True,
        ),
    ]

    # Full-volume inputs remain variable-sized. The CPU stage only performs
    # the configured fractional center crop and normalization.
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
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=file_store.test,
        test_transforms=cpu_val_transforms,
        use_random_datasampler=False,
        use_weighted_sampler=cfg.training.use_weighted_sampler,
        weighted_sampler_power=cfg.training.weighted_sampler_power,
        train_num_samples=cfg.training.samples_per_epoch,
        sampler_seed=cfg.training.seed,
    )

    num_classes = int(file_store.dataset_json["metadata"]["n_classes"])
    model = instantiate(cfg.model._cls_net, num_classes=num_classes)

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        weights=pretrained_weights,
        min_backbone_load_fraction=cfg.model.min_backbone_load_fraction,
        trainable_backbone_blocks=cfg.model.trainable_backbone_blocks,
        learning_rate=cfg.model.finetune_lr,
        backbone_lr_multiplier=cfg.model.backbone_lr_multiplier,
        minimum_lr=cfg.model.minimum_lr,
        warmup_ratio=cfg.model.warmup_ratio,
        cosine_period_ratio=cfg.model.cosine_period_ratio,
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
        limit_val_batches=cfg.training.val_steps_per_epoch,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        use_distributed_sampler=False,
        num_sanity_val_steps=cfg.training.num_sanity_val_steps,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=cfg.training.resume_ckpt,
    )

    if file_store.test:
        trainer.test(
            model=model_module,
            datamodule=data_module,
            ckpt_path="best",
        )


if __name__ == "__main__":
    main()
