from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Mapping, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from asparagus.functional.lr_scheduling import (
    cosine_decay_schedule,
    simple_warmup_cosine_decay_schedule,
)
from asparagus.functional.visualization import (
    get_logger_compatible_image_output_target,
    log_image_output_target_to_mlflow,
    log_image_output_target_to_wandb,
)
from asparagus.modules.networks.vit_task_model import ViTTaskModel
from torch.optim import AdamW, SGD
from torchvision import transforms


class ViTTaskBaseModule(L.LightningModule):
    """Lightning base for sliding-window ViT task models.

    This class deliberately does not override ``load_state_dict``. Lightning
    can therefore restore a downstream-training checkpoint normally. Transfer
    from a BrainDINO checkpoint is handled separately by
    ``load_pretrained_weights`` and loads only the pretrained backbone.
    """

    def __init__(
            self,
            model: ViTTaskModel,
            learning_rate: float = 1e-4,

            # Retained only because BaseModule expects it.
            warmup_epochs: int = 0,

            warmup_ratio: float = 0.10,
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

        if not 0.0 < warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in (0, 1).")

        if not 0.0 < cosine_period_ratio <= 1.0:
            raise ValueError(
                "cosine_period_ratio must be in (0, 1]."
            )

        if minimum_lr <= 0.0:
            raise ValueError("minimum_lr must be positive.")

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
        self.warmup_ratio = float(warmup_ratio)
        self.cosine_period_ratio = float(
            cosine_period_ratio
        )
        self.minimum_lr = float(minimum_lr)

        self.optimizer_name = optimizer
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.log_image_every_n_epochs = int(
            log_image_every_n_epochs
        )
        self.test_output_path = test_output_path

        self.loss: Optional[nn.Module] = None
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

        if weights is not None:
            self.load_pretrained_weights(
                weights,
                min_load_fraction=min_backbone_load_fraction,
            )

        if compile_mode is not None:
            self.model = torch.compile(
                self.model,
                mode=compile_mode,
            )

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
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(
                "ViTTaskModel requires image [1, C, H, W, D]; "
                f"received {tuple(x.shape)}. Set batch_size=1."
            )

        info = batch["info"]
        return self.model(
            x=x,
            spacing=info["spacing"],
            modality=info.get("modality"),
            channel_mask=info.get("channel_mask"),
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
        """Load only SpacingAwareViT3d weights from a BrainDINO checkpoint."""
        if not 0.0 < min_load_fraction <= 1.0:
            raise ValueError("min_load_fraction must be in (0, 1].")

        source = self._unwrap_checkpoint_state(checkpoint)
        backbone = self.unwrap_compiled_model().backbone
        target = backbone.state_dict()
        mapped = {}
        mapped_priority = {}
        wrong_shape = []

        for original_key, value in source.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = self._remove_compile_prefix(original_key)

            # A full SSL checkpoint may contain both networks. Downstream
            # checkpoints are initialized from the saved student, never allow a
            # teacher entry encountered later to overwrite it.
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
                    (backbone_key, tuple(value.shape), tuple(target[backbone_key].shape))
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
                "Only "
                f"{len(mapped)}/{len(target)} ({loaded_fraction:.1%}) backbone "
                "entries matched the BrainDINO checkpoint. "
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

    def _parameter_groups(self):
        model = self.unwrap_compiled_model()

        backbone_parameters = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]

        backbone_parameter_ids = {
            id(parameter)
            for parameter in backbone_parameters
        }

        task_parameters = [
            parameter
            for parameter in model.parameters()
            if (
                    parameter.requires_grad
                    and id(parameter)
                    not in backbone_parameter_ids
            )
        ]

        groups = []

        if task_parameters:
            groups.append(
                {
                    "params": task_parameters,
                    "lr": self.learning_rate,
                    "name": "task_head",
                }
            )

        if backbone_parameters:
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": self.learning_rate,
                    "name": "backbone",
                }
            )

        if not groups:
            raise RuntimeError(
                "The model has no trainable parameters."
            )

        return groups

    def configure_optimizers(self):
        parameter_groups = self._parameter_groups()

        optimizer_name = self.optimizer_name.lower()

        if optimizer_name == "adamw":
            optimizer = AdamW(
                parameter_groups,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )

        elif optimizer_name == "sgd":
            optimizer = SGD(
                parameter_groups,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )

        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}."
            )

        # This already accounts for epochs, batch limits,
        # gradient accumulation, devices, and max_steps.
        total_steps = int(
            self.trainer.estimated_stepping_batches
        )

        if total_steps <= 0:
            raise RuntimeError(
                "Could not determine a positive number of "
                f"optimizer steps: {total_steps}."
            )

        scheduler = (
            simple_warmup_cosine_decay_schedule(
                optimizer=optimizer,
                total_steps=total_steps,
                warmup_ratio=self.warmup_ratio,
                cosine_period_ratio=(
                    self.cosine_period_ratio
                ),
                minimum_lr=self.minimum_lr,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.train_transforms is not None:
            batch = self.train_transforms(batch)
        elif (
            self.trainer.validating or self.trainer.sanity_checking
        ) and self.val_transforms is not None:
            batch = self.val_transforms(batch)
        elif (
            self.trainer.testing or self.trainer.predicting
        ) and self.test_transforms is not None:
            batch = self.test_transforms(batch)
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
        file_name = imagedict["file"][batch_index].split("/Task")[-1]

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
