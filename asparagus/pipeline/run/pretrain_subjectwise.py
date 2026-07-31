import hydra
import lightning as pl
import random
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import PretrainSearchpathPlugin
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_ssl_plugins, prepare_subjectwise_experiment
from asparagus.pipeline.auto_configuration.logging import logging
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, ThroughputMonitor
from omegaconf import DictConfig, OmegaConf
from asparagus.modules.networks.blocks.utils import to_3tuple
import math

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("version", lambda: generate_unused_run_id(), use_cache=True)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ceil_div",
    lambda numerator, denominator: (int(numerator) + int(denominator) - 1)
    // int(denominator),
)
Plugins.instance().register(PretrainSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    if HydraConfig.get().runtime.output_dir:
        print(f"Version: {cfg.run_id}")
        print(f"Run dir: {HydraConfig.get().run.dir}")

    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_subjectwise_experiment(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    plugins = prepare_ssl_plugins(cfg)

    assert cfg.task is not None, "Config file is not set up correctly."

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
        TQDMProgressBar(
            refresh_rate=cfg.logger.log_every_n_steps,
        ),

        # Latest full checkpoint for resuming interrupted training.
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            filename="recovery-step-{step}",
            every_n_train_steps=(
                    cfg.model.recovery_ckpt_every_n_optimizer_steps
                    * cfg.training.accumulate_grad_batches
            ),
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            enable_version_counter=False,
        ),

        # Best checkpoint according to validation loss.
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            filename="best-step-{step}-val_loss-{val/loss/total:.4f}",
            monitor="val/loss/total",
            mode="min",
            save_top_k=1,
            save_last=False,
            save_weights_only=False,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        ),

        # Permanent snapshots for downstream evaluation.
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            filename="snapshot-step-{step}",
            every_n_train_steps=(
                    cfg.model.snapshot_ckpt_every_n_optimizer_steps
                    * cfg.training.accumulate_grad_batches
            ),
            save_top_k=-1,
            save_last=False,
            save_weights_only=False,
            enable_version_counter=False,
        ),

        LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
            log_weight_decay=True,
        ),
    ] + plugins

    if cfg.profiler.enabled:
        callbacks.append(
            instantiate(cfg.profiler._callback)
        )

    cpu_tr_transforms = instantiate(
        cfg.transforms._cpu_tr_transforms,
        source_patch_size=cfg.training.source_patch_size,
        normalize=cfg.transforms.normalize,
    )

    cpu_val_transforms = instantiate(
        cfg.transforms._cpu_val_transforms,
        source_patch_size=cfg.training.source_patch_size,
        normalize=cfg.transforms.normalize,
    )

    gpu_tr_transforms = instantiate(
        cfg.transforms._gpu_tr_transforms,
        global_crop_size=cfg.training.global_crop_size,
        local_crop_size=cfg.training.local_crop_size,
        token_stride=to_3tuple(cfg.model.patch_stride),
        mask_ratio=cfg.training.mask_ratio,
        n_global_crops=cfg.training.n_global_crops,
        n_local_crops=cfg.training.n_local_crops,
    )

    gpu_val_transforms = instantiate(
        cfg.transforms._gpu_val_transforms,
        global_crop_size=cfg.training.global_crop_size,
        local_crop_size=cfg.training.local_crop_size,
        token_stride=to_3tuple(cfg.model.patch_stride),
        mask_ratio=cfg.training.mask_ratio,
        n_global_crops=cfg.training.n_global_crops,
        n_local_crops=cfg.training.n_local_crops,
    )

    model = instantiate(
        cfg.model._pretrain_net,
    )

    data_module_factory = instantiate(
        cfg.lightning._data_module,
        _partial_=True,
    )
    data_module = data_module_factory(
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        learning_rate=cfg.model.pretrain_lr,

        # BaseModule compatibility only; not used by BrainDINO scheduler.
        warmup_epochs=0,

        warmup_ratio=cfg.model.warmup_ratio,
        cosine_period_ratio=cfg.model.cosine_period_ratio,
        minimum_lr=cfg.model.minimum_lr,

        train_transforms=gpu_tr_transforms,
        val_transforms=gpu_val_transforms,
        optimizer=cfg.model.pretrain_optim,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        weight_decay=cfg.model.weight_decay,
    )

    trainer = instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        default_root_dir=path_store.run_dir,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        max_steps=cfg.training.steps,
        limit_train_batches=cfg.training.steps_per_epoch,
        limit_val_batches=cfg.training.val_steps_per_epoch,
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        num_sanity_val_steps=2,
    )

    if trainer.is_global_zero:
        physical_global_batch_size = cfg.training.batch_size * cfg.hardware.num_devices * cfg.hardware.num_nodes
        effective_global_batch_size = physical_global_batch_size * cfg.training.accumulate_grad_batches
        optimizer_steps_per_pseudo_epoch = math.ceil(cfg.training.steps_per_epoch / cfg.training.accumulate_grad_batches)

        pseudo_epochs = cfg.training.steps / optimizer_steps_per_pseudo_epoch

        samples_per_pseudo_epoch = cfg.training.steps_per_epoch * physical_global_batch_size

        warmup_steps = round(cfg.training.steps * cfg.model.warmup_ratio)

        print("Training duration configured as:")
        print(f"  - Maximum optimizer steps: {cfg.training.steps:,}")
        print(f"  - Physical global batch size: {physical_global_batch_size}")
        print(f"  - Effective global batch size: {effective_global_batch_size}")
        print(f"  - Trainer batches per pseudo-epoch: {cfg.training.steps_per_epoch:,}")
        print(f"  - Optimizer steps per pseudo-epoch: {optimizer_steps_per_pseudo_epoch:,}")
        print(f"  - Samples per pseudo-epoch: {samples_per_pseudo_epoch:,}")
        print(f"  - Expected pseudo-epochs: {pseudo_epochs:.1f}")
        print(f"  - Warmup: {warmup_steps:,} optimizer steps ({cfg.model.warmup_ratio:.1%})")

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
