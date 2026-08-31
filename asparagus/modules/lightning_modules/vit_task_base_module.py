from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import Any, Mapping, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from asparagus.functional.visualization import (
    get_logger_compatible_image_output_target,
    log_image_output_target_to_mlflow,
    log_image_output_target_to_wandb,
)
from asparagus.modules.networks.vit_task_model import ViTTaskModel
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms


class ViTTaskBaseModule(L.LightningModule):
    """Shared Lightning functionality for downstream ViT tasks.

    BrainDINO checkpoints initialize only the student backbone. A downstream
    checkpoint is still restored normally by Lightning because this class does
    not override ``load_state_dict``.

    The learning-rate schedule is defined directly in optimizer-step units:

    1. optional linear warmup for ``warmup_ratio * total_steps``;
    2. cosine decay for ``cosine_period_ratio`` of the remaining steps;
    3. constant minimum learning rate for any remaining steps.

    The same multiplicative schedule is applied to the task-head and backbone
    parameter groups, preserving their configured learning-rate ratio.
    """

    def __init__(
        self,
        model: ViTTaskModel,
        learning_rate: float = 1e-4,
        backbone_learning_rate: Optional[float] = None,
        warmup_ratio: float = 0.1,
        cosine_period_ratio: float = 1.0,
        minimum_lr: float = 1e-6,
        optimizer: str = "AdamW",
        weight_decay: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        compile_mode: Optional[str] = None,
        weights: Optional[Mapping[str, Any]] = None,
        min_backbone_load_fraction: float = 0.95,
        train_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        log_image_every_n_epochs: int = 50,
        test_output_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if not isinstance(model, ViTTaskModel):
            raise TypeError(
                "model must inherit ViTTaskModel, got "
                f"{type(model).__name__}."
            )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if backbone_learning_rate is not None and backbone_learning_rate < 0.0:
            raise ValueError("backbone_learning_rate must be non-negative when provided.")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if not 0.0 < cosine_period_ratio <= 1.0:
            raise ValueError("cosine_period_ratio must be in (0, 1].")
        if minimum_lr < 0.0:
            raise ValueError("minimum_lr must be non-negative.")
        if minimum_lr > learning_rate:
            raise ValueError(
                "minimum_lr cannot exceed the task-head learning rate: "
                f"minimum_lr={minimum_lr}, learning_rate={learning_rate}."
            )
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0.")
        if log_image_every_n_epochs < 0:
            raise ValueError("log_image_every_n_epochs must be non-negative.")

        self.save_hyperparameters(
            ignore=(
                "model",
                "weights",
                "train_transforms",
                "val_transforms",
                "test_transforms",
            )
        )

        self.model = model
        self.learning_rate = float(learning_rate)
        self.backbone_learning_rate = (
            float(backbone_learning_rate)
            if backbone_learning_rate is not None
            else self.learning_rate * 0.1
        )
        self.warmup_ratio = float(warmup_ratio)
        self.cosine_period_ratio = float(cosine_period_ratio)
        self.minimum_lr = float(minimum_lr)
        self.optimizer_name = str(optimizer)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.log_image_every_n_epochs = int(log_image_every_n_epochs)
        self.test_output_path = test_output_path

        self.loss: Optional[nn.Module] = None
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

        # Transfer before compilation so parameter names remain predictable.
        if weights is not None:
            self.load_pretrained_weights(
                weights,
                min_load_fraction=min_backbone_load_fraction,
            )

        if compile_mode is not None:
            self.model = torch.compile(self.model, mode=compile_mode)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def unwrap_compiled_model(self) -> ViTTaskModel:
        model = self.model
        while hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def forward_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        x = batch["image"]
        if x.ndim != 5:
            raise ValueError(
                "ViTTaskModel requires image [B, C, H, W, D]; "
                f"received {tuple(x.shape)}."
            )

        info = batch["info"]
        return self.model(
            x=x,
            spacing=info["spacing"],
            modality=info.get("modality"),
            channel_mask=info.get("channel_mask"),
            valid_spatial_shapes=info.get("valid_spatial_shapes"),
        )

    @staticmethod
    def _unwrap_checkpoint_state(
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("Checkpoint state_dict must be a mapping.")
        return state

    @staticmethod
    def _remove_compile_prefix(key: str) -> str:
        return key.replace("_orig_mod.", "")

    def load_pretrained_weights(
        self,
        checkpoint: Mapping[str, Any],
        min_load_fraction: float = 0.95,
    ) -> None:
        """Load only the pretrained student SpacingAwareViT3d backbone."""
        if not 0.0 < min_load_fraction <= 1.0:
            raise ValueError("min_load_fraction must be in (0, 1].")

        source = self._unwrap_checkpoint_state(checkpoint)
        backbone = self.unwrap_compiled_model().backbone
        target = backbone.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        mapped_priority: dict[str, int] = {}
        wrong_shape = []

        for original_key, value in source.items():
            if not isinstance(original_key, str) or not isinstance(value, torch.Tensor):
                continue

            key = self._remove_compile_prefix(original_key)
            if ".teacher." in key or key.startswith("teacher."):
                continue

            priority = 2 if (".student." in key or key.startswith("student.")) else 1
            if ".backbone." in key:
                backbone_key = key.split(".backbone.", maxsplit=1)[1]
            elif key.startswith("backbone."):
                backbone_key = key.removeprefix("backbone.")
            else:
                continue

            if backbone_key not in target:
                continue
            if target[backbone_key].shape != value.shape:
                wrong_shape.append(
                    (
                        backbone_key,
                        tuple(value.shape),
                        tuple(target[backbone_key].shape),
                    )
                )
                continue
            if priority < mapped_priority.get(backbone_key, 0):
                continue

            mapped[backbone_key] = value
            mapped_priority[backbone_key] = priority

        missing = sorted(set(target) - set(mapped))
        loaded_fraction = len(mapped) / max(1, len(target))
        if loaded_fraction < min_load_fraction:
            raise RuntimeError(
                f"Only {len(mapped)}/{len(target)} ({loaded_fraction:.1%}) "
                "backbone entries matched the BrainDINO checkpoint. "
                f"First missing keys: {missing[:10]}; "
                f"first shape mismatches: {wrong_shape[:5]}."
            )

        incompatible = backbone.load_state_dict(mapped, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected mapped backbone keys: {incompatible.unexpected_keys}."
            )

        logging.info(
            "Loaded %d/%d (%.1f%%) pretrained ViT backbone entries. "
            "The downstream task head remains newly initialized.",
            len(mapped),
            len(target),
            100.0 * loaded_fraction,
        )
        if missing:
            logging.warning("Missing pretrained backbone entries: %s", missing)

    def _parameter_groups(self) -> list[dict[str, Any]]:
        model = self.unwrap_compiled_model()
        backbone_parameters = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]

        groups: list[dict[str, Any]] = []
        if head_parameters:
            groups.append(
                {
                    "params": head_parameters,
                    "lr": self.learning_rate,
                    "name": "task_head",
                }
            )
        if backbone_parameters:
            if self.backbone_learning_rate <= 0.0:
                raise ValueError(
                    "backbone_learning_rate must be positive when backbone "
                    "parameters are trainable."
                )
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": self.backbone_learning_rate,
                    "name": "backbone",
                }
            )
        if not groups:
            raise RuntimeError("The model has no trainable parameters.")

        logging.info(
            "Trainable parameters: head=%d, backbone=%d; learning rates: "
            "head=%.3e, backbone=%.3e",
            sum(parameter.numel() for parameter in head_parameters),
            sum(parameter.numel() for parameter in backbone_parameters),
            self.learning_rate,
            self.backbone_learning_rate,
        )
        return groups

    def _make_optimizer(self, groups: list[dict[str, Any]]):
        name = self.optimizer_name.lower()
        if name == "adamw":
            return AdamW(
                groups,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
        if name == "sgd":
            return SGD(
                groups,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        raise ValueError(
            f"Unknown optimizer {self.optimizer_name!r}; expected AdamW or SGD."
        )

    def _lr_multiplier(self, step: int, total_steps: int) -> float:
        """Return the shared head/backbone LR multiplier for one step."""
        if total_steps < 1:
            raise ValueError("total_steps must be positive.")

        # minimum_lr is specified for the task head. Applying the corresponding
        # ratio to every group preserves the backbone/head LR relationship.
        minimum_multiplier = self.minimum_lr / self.learning_rate
        warmup_steps = min(
            total_steps - 1,
            int(round(total_steps * self.warmup_ratio)),
        )

        if warmup_steps > 0 and step < warmup_steps:
            return max(minimum_multiplier, (step + 1) / warmup_steps)

        remaining_steps = max(1, total_steps - warmup_steps)
        cosine_steps = max(
            1,
            int(round(remaining_steps * self.cosine_period_ratio)),
        )
        cosine_step = min(max(0, step - warmup_steps), cosine_steps)
        progress = cosine_step / cosine_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_multiplier + (1.0 - minimum_multiplier) * cosine

    def configure_optimizers(self):
        optimizer = self._make_optimizer(self._parameter_groups())
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps < 1:
            raise RuntimeError(
                "Lightning estimated no optimizer steps. Check the training "
                "dataloader, limit_train_batches, max_epochs, and gradient "
                "accumulation settings."
            )

        warmup_steps = int(round(total_steps * self.warmup_ratio))
        logging.info(
            "LR schedule: total_steps=%d, warmup_steps=%d, "
            "cosine_period_ratio=%.3f, minimum_head_lr=%.3e",
            total_steps,
            warmup_steps,
            self.cosine_period_ratio,
            self.minimum_lr,
        )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._lr_multiplier(step, total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.train_transforms is not None:
            return self.train_transforms(batch)
        if (
            self.trainer.validating or self.trainer.sanity_checking
        ) and self.val_transforms is not None:
            return self.val_transforms(batch)
        if (
            self.trainer.testing or self.trainer.predicting
        ) and self.test_transforms is not None:
            return self.test_transforms(batch)
        return batch

    def _log_dict_of_images_to_wandb(
        self,
        imagedict: dict,
        log_key: str,
        task_type: str = "",
    ) -> None:
        batch_index = np.random.randint(0, imagedict["input"].shape[0])
        image, output, target = get_logger_compatible_image_output_target(
            image=imagedict["input"][batch_index],
            output=imagedict["output"][batch_index],
            target=imagedict["target"][batch_index],
            task_type=task_type,
        )

        file_value = imagedict["file"][batch_index]
        file_name = str(file_value).split("/Task")[-1]
        for logger in self.trainer.loggers:
            if "WandbLogger" in logger.__class__.__name__:
                log_image_output_target_to_wandb(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=file_name,
                    step=self.global_step,
                    task_type=task_type,
                )
            elif "MLFlowLogger" in logger.__class__.__name__:
                log_image_output_target_to_mlflow(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=file_name,
                    step=self.global_step,
                    task_type=task_type,
                )
