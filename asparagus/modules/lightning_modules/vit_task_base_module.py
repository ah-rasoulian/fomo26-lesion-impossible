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
        backbone_learning_rate: Optional[float] = None,
        warmup_epochs: int = 3,
        cosine_period_ratio: float = 1.0,
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
        if not 0.0 < cosine_period_ratio <= 1.0:
            raise ValueError("cosine_period_ratio must be in (0, 1].")

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
        self.warmup_epochs = int(warmup_epochs or 0)
        self.cosine_period_ratio = float(cosine_period_ratio)
        self.optimizer_name = optimizer
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

        # Load before torch.compile so checkpoint names are stable and simple.
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
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]

        groups = []
        if head_parameters:
            groups.append(
                {
                    "params": head_parameters,
                    "lr": self.learning_rate,
                    "name": "task_head",
                }
            )
        if backbone_parameters:
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": self.backbone_learning_rate,
                    "name": "backbone",
                }
            )
        if not groups:
            raise RuntimeError("The model has no trainable parameters.")
        return groups

    def configure_optimizers(self):
        groups = self._parameter_groups()
        name = self.optimizer_name.lower()
        if name == "adamw":
            optimizer = AdamW(
                groups,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
        elif name == "sgd":
            optimizer = SGD(
                groups,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}.")

        if self.trainer.max_epochs > 0:
            steps_per_epoch = max(
                1,
                self.trainer.estimated_stepping_batches
                // self.trainer.max_epochs,
            )
        else:
            limit = self.trainer.limit_train_batches
            if not isinstance(limit, int):
                raise ValueError(
                    "When training with max_steps, limit_train_batches must "
                    "be an integer so steps_per_epoch can be determined."
                )
            steps_per_epoch = max(
                1,
                limit // self.trainer.accumulate_grad_batches,
            )

        if self.warmup_epochs > 0:
            scheduler = simple_warmup_cosine_decay_schedule(
                optimizer,
                self.warmup_epochs,
                steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,
                self.trainer.max_steps,
            )
        else:
            scheduler = cosine_decay_schedule(
                optimizer,
                steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,
                self.trainer.max_steps,
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
