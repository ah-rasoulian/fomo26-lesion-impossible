from __future__ import annotations

import ast
import math
import operator
import random
from typing import Any

import hydra
import lightning as pl
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

from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import (
    PretrainSearchpathPlugin,
)
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.experiment_setup import (
    prepare_ssl_plugins,
    prepare_subjectwise_experiment,
)
from asparagus.pipeline.auto_configuration.logging import logging


load_dotenv()


def _multiply(*values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_SAFE_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "float": float,
    "floor": math.floor,
    "int": int,
    "max": max,
    "min": min,
    "round": round,
}


def _evaluate_arithmetic_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic_node(node.left)
        right = _evaluate_arithmetic_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 16:
            raise ValueError("Exponent is too large in eval resolver.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](
            _evaluate_arithmetic_node(node.operand)
        )
    if isinstance(node, ast.BoolOp) and isinstance(
        node.op, (ast.And, ast.Or)
    ):
        values = [_evaluate_arithmetic_node(value) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _evaluate_arithmetic_node(node.left)
        for operation, comparator in zip(node.ops, node.comparators):
            function = _COMPARISON_OPERATORS.get(type(operation))
            if function is None:
                raise ValueError("Unsupported comparison in eval resolver.")
            right = _evaluate_arithmetic_node(comparator)
            if not function(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        condition = _evaluate_arithmetic_node(node.test)
        return _evaluate_arithmetic_node(
            node.body if condition else node.orelse
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_FUNCTIONS
        and not node.keywords
    ):
        arguments = [_evaluate_arithmetic_node(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[node.func.id](*arguments)
    raise ValueError(
        "The eval resolver accepts only numeric arithmetic, comparisons, "
        "boolean operations, and conditional expressions."
    )


def _safe_arithmetic_eval(expression: Any) -> Any:
    """Backward-compatible, non-executable replacement for config eval."""
    if isinstance(expression, (int, float, bool)):
        return expression
    tree = ast.parse(str(expression), mode="eval")
    return _evaluate_arithmetic_node(tree)


def _register_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "random",
        lambda low, high: random.randint(int(low), int(high)),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "version",
        lambda: generate_unused_run_id(),
        use_cache=True,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "multiply",
        _multiply,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "floor_div",
        lambda numerator, denominator: int(numerator) // int(denominator),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "ceil_div",
        lambda numerator, denominator: (
            int(numerator) + int(denominator) - 1
        )
        // int(denominator),
        replace=True,
    )
    # Older shared Hydra configs still use ${eval:"a // b"}. Keep that
    # resolver name, but permit only a small arithmetic expression grammar.
    OmegaConf.register_new_resolver(
        "eval",
        _safe_arithmetic_eval,
        replace=True,
    )


_register_resolvers()
Plugins.instance().register(PretrainSearchpathPlugin)


def _validate_configuration(cfg: DictConfig) -> None:
    grid_size = tuple(int(value) for value in cfg.model.grid_size)
    token_grid = tuple(int(value) for value in cfg.training.token_grid_size)
    if grid_size != token_grid:
        raise ValueError(
            "model.grid_size and training.token_grid_size must match: "
            f"{grid_size} versus {token_grid}."
        )
    if any(value < 1 for value in grid_size):
        raise ValueError("The token-grid dimensions must be positive.")
    if int(cfg.model.patch_kernel_size) != 8:
        raise ValueError("The configured physical patch kernel must be 8.")
    if int(cfg.training.n_global_crops) < 2:
        raise ValueError("BrainDINO requires at least two global views.")
    if int(cfg.training.batch_size) < 2:
        raise ValueError(
            "Per-device batch_size must be at least 2 because KoLeo computes "
            "nearest neighbours within each forward pass; gradient "
            "accumulation does not increase that set."
        )
    if int(cfg.training.n_local_crops) > 0 and (
        cfg.training.local_crop_size is None
    ):
        raise ValueError(
            "Set local_crop_size when n_local_crops is positive, or use zero "
            "local crops for full-volume-only pretraining."
        )
    expected_batches = math.ceil(
        int(cfg.training.samples_per_epoch)
        / int(cfg.training.physical_global_batch_size)
    )
    if int(cfg.training.steps_per_epoch) != expected_batches:
        raise ValueError(
            "steps_per_epoch is inconsistent with samples_per_epoch and "
            "physical_global_batch_size."
        )


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    _validate_configuration(cfg)

    if HydraConfig.get().runtime.output_dir:
        print(f"Version: {cfg.run_id}")
        print(f"Run dir: {HydraConfig.get().run.dir}")

    logging_safe_cfg = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    file_store, path_store, version_store = prepare_subjectwise_experiment(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)
    plugins = prepare_ssl_plugins(cfg)

    if cfg.task is None:
        raise ValueError("The task configuration is missing.")

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
        TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps),
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            filename="recovery-step-{step}",
            every_n_train_steps=(
                cfg.model.recovery_ckpt_every_n_optimizer_steps
            ),
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            enable_version_counter=False,
        ),
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
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            filename="snapshot-step-{step}",
            every_n_train_steps=(
                cfg.model.snapshot_ckpt_every_n_optimizer_steps
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
        callbacks.append(instantiate(cfg.profiler._callback))

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

    common_gpu_transform_arguments = {
        "token_grid_size": cfg.training.token_grid_size,
        "local_crop_size": cfg.training.local_crop_size,
        "mask_ratio": cfg.training.mask_ratio,
        "min_mask_block_fraction": cfg.training.min_mask_block_fraction,
        "max_mask_block_fraction": cfg.training.max_mask_block_fraction,
        "n_global_crops": cfg.training.n_global_crops,
        "n_local_crops": cfg.training.n_local_crops,
    }
    gpu_tr_transforms = instantiate(
        cfg.transforms._gpu_tr_transforms,
        **common_gpu_transform_arguments,
        channel_drop_probability=cfg.transforms.channel_drop_probability,
        one_unknown_probability=cfg.transforms.one_unknown_probability,
        all_unknown_probability=cfg.transforms.all_unknown_probability,
        unknown_modality_id=cfg.transforms.unknown_modality_id,
        local_center_jitter_fraction=cfg.transforms.local_center_jitter_fraction,
        local_min_foreground_fraction=cfg.transforms.local_min_foreground_fraction,
        local_crop_attempts=cfg.transforms.local_crop_attempts,
        flip_probability=cfg.transforms.flip_probability,
        affine_probability=cfg.transforms.affine_probability,
        rotation_degrees=cfg.transforms.rotation_degrees,
        scale_range=cfg.transforms.scale_range,
        translation_fraction=cfg.transforms.translation_fraction,
    )
    gpu_val_transforms = instantiate(
        cfg.transforms._gpu_val_transforms,
        **common_gpu_transform_arguments,
        local_center_jitter_fraction=0.0,
        local_min_foreground_fraction=cfg.transforms.local_min_foreground_fraction,
        local_crop_attempts=1,
    )

    model = instantiate(cfg.model._pretrain_net)

    data_module_factory = instantiate(
        cfg.lightning._data_module,
        _partial_=True,
    )
    data_module = data_module_factory(
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        batch_size=cfg.training.batch_size,
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        num_samples=cfg.training.samples_per_epoch,
        max_channels=cfg.training.max_channels,
        unknown_modality_id=cfg.transforms.unknown_modality_id,
        train_metadata_cache_path=cfg.data.train_metadata_cache_path,
        val_metadata_cache_path=cfg.data.val_metadata_cache_path,
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        learning_rate=cfg.model.pretrain_lr,
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
        visual_log_every_n_steps=cfg.logger.visual_log_every_n_steps,
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
        num_sanity_val_steps=cfg.training.num_sanity_val_steps,
    )

    if trainer.is_global_zero:
        physical_batch = int(cfg.training.physical_global_batch_size)
        effective_batch = int(cfg.training.effective_global_batch_size)
        optimizer_steps_per_epoch = math.ceil(
            int(cfg.training.steps_per_epoch)
            / int(cfg.training.accumulate_grad_batches)
        )
        pseudo_epochs = cfg.training.steps / optimizer_steps_per_epoch
        samples_per_pseudo_epoch = (
            int(cfg.training.steps_per_epoch) * physical_batch
        )
        warmup_steps = round(cfg.training.steps * cfg.model.warmup_ratio)

        print("Training duration configured as:")
        print(f"  - Maximum optimizer steps: {cfg.training.steps:,}")
        print(f"  - Physical global batch size: {physical_batch}")
        print(f"  - Effective global batch size: {effective_batch}")
        print(
            "  - Trainer batches per pseudo-epoch: "
            f"{cfg.training.steps_per_epoch:,}"
        )
        print(
            "  - Optimizer steps per pseudo-epoch: "
            f"{optimizer_steps_per_epoch:,}"
        )
        print(f"  - Samples per pseudo-epoch: {samples_per_pseudo_epoch:,}")
        print(f"  - Expected pseudo-epochs: {pseudo_epochs:.1f}")
        print(
            f"  - Warmup: {warmup_steps:,} optimizer steps "
            f"({cfg.model.warmup_ratio:.1%})"
        )

    resume_ckpt = cfg.training.get("resume_ckpt")
    if resume_ckpt:
        print("Resuming from checkpoint:", resume_ckpt)
    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    main()
