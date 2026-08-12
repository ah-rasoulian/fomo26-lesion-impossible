from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.modules.lightning_modules.vit_task_base_module import (
    ViTTaskBaseModule,
)
from gardening_tools.functional.paths.write import save_json
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


class ViTClsRegModule(ViTTaskBaseModule):
    """Shared Lightning logic for ViT classification and regression tasks."""

    def __init__(
        self,
        *args,
        log_image_every_n_epochs: int = 50,
        test_output_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            log_image_every_n_epochs=log_image_every_n_epochs,
            test_output_path=test_output_path,
            **kwargs,
        )

        task_model = self.unwrap_compiled_model()
        self.num_outputs = self._infer_num_outputs(task_model)
        # Retain this name for compatibility with the existing configs/code.
        self.num_classes = self.num_outputs
        self.ignore_index_in_metrics = -1
        self.task_type = ""
        self.loss: nn.Module

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_test_metrics()

        self.results: dict[str, Any] = {}
        self.predictions: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    @staticmethod
    def _infer_num_outputs(model: nn.Module) -> int:
        if hasattr(model, "num_classes"):
            return int(model.num_classes)
        if hasattr(model, "output_dim"):
            return int(model.output_dim)

        # Supports the current _GlobalTaskHead implementation.
        for head_name in ("classifier", "regressor"):
            head = getattr(model, head_name, None)
            if head is None:
                continue
            for layer in reversed(list(head.modules())):
                if isinstance(layer, nn.Linear):
                    return int(layer.out_features)

        raise AttributeError(
            "The task model must expose num_classes/output_dim or contain "
            "a Linear classifier/regressor output layer."
        )

    @abstractmethod
    def prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def configure_metrics(self, prefix: str) -> MetricCollection:
        raise NotImplementedError

    @abstractmethod
    def configure_test_metrics(self) -> MetricCollection:
        raise NotImplementedError

    @abstractmethod
    def _record_test_batch(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    def _shared_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        target = self.prepare_target(batch["CLSREG_label"])
        outputs = self.forward_batch(batch)
        loss = self.loss(outputs, target)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(outputs.detach(), target.detach())

        if (
            self.current_epoch > 0
            and batch_idx == 0
            and self.log_image_every_n_epochs > 0
            and self.current_epoch % self.log_image_every_n_epochs == 0
            and self.logger is not None
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": batch["image"].detach().float().cpu().numpy(),
                    "target": target.detach().float().cpu().numpy(),
                    "output": outputs.detach().float().cpu().numpy(),
                    "file": batch["file_path"],
                },
                log_key=stage,
                task_type=self.task_type,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def _finish_metric_epoch(
        self,
        metrics: MetricCollection,
    ) -> None:
        values = metrics.compute()
        values = format_multilabel_metrics(
            values,
            ignore_index=self.ignore_index_in_metrics,
        )
        self.log_dict(values, sync_dist=True)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._finish_metric_epoch(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._finish_metric_epoch(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        self.results = {}
        self.predictions = []
        self.labels = []
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.forward_batch(batch)
        target = self.prepare_target(batch["CLSREG_label"])
        self.test_metrics.update(outputs.detach(), target.detach())
        self._record_test_batch(outputs, target, batch)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward_batch(batch)

    @staticmethod
    def _file_key(batch: Mapping[str, Any]) -> str:
        file_path = batch["file_path"]
        if isinstance(file_path, (list, tuple)):
            return str(file_path[0])
        return str(file_path)

    def on_test_epoch_end(self) -> None:
        metric_values = self.test_metrics.compute()
        metric_values = {
            key: value.detach().cpu().tolist()
            for key, value in metric_values.items()
        }
        self.results["metrics"] = metric_values

        if self.test_output_path is not None:
            output_directory = os.path.dirname(self.test_output_path)
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
            save_json(self.results, self.test_output_path)

        logging.info(
            "Aggregated test results for %d cases: %s",
            len(self.predictions),
            metric_values,
        )
        self.test_metrics.reset()


class ViTClassificationModule(ViTClsRegModule):
    """Multiclass classification, including two-logit binary models."""

    def __init__(
        self,
        *args,
        label_smoothing: float = 0.0,
        loss_weight: Optional[Sequence[float]] = None,
        positive_class_index: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_classes < 2:
            raise ValueError(
                "ViTClassificationModule uses CrossEntropyLoss and therefore "
                "requires at least two output logits."
            )
        if not 0 <= positive_class_index < self.num_classes:
            raise ValueError("positive_class_index is outside the class range.")

        self.positive_class_index = int(positive_class_index)
        weight = (
            torch.as_tensor(loss_weight, dtype=torch.float32)
            if loss_weight is not None
            else None
        )
        if weight is not None and weight.numel() != self.num_classes:
            raise ValueError(
                f"loss_weight must contain {self.num_classes} values."
            )
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )
        self.task_type = "classification"

    def prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        return target.reshape(-1).long()

    def configure_metrics(self, prefix: str) -> MetricCollection:
        return MetricCollection(
            {
                f"{prefix}/acc": MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                f"{prefix}/auroc": MulticlassAUROC(
                    num_classes=self.num_classes,
                    average="macro",
                ),
            }
        )

    def configure_test_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "AUROC": MulticlassAUROC(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "Precision": MulticlassPrecision(
                    num_classes=self.num_classes,
                    average=None,
                ),
                "Recall": MulticlassRecall(
                    num_classes=self.num_classes,
                    average=None,
                ),
            }
        )

    def _record_test_batch(self, outputs, target, batch) -> None:
        probabilities = outputs.softmax(dim=-1)
        prediction = outputs.argmax(dim=-1)
        key = self._file_key(batch)

        record = {
            "prediction": int(prediction[0].detach().cpu()),
            "probabilities": probabilities[0].detach().cpu().tolist(),
            "label": int(target[0].detach().cpu()),
        }
        if self.num_classes == 2:
            record["probability"] = float(
                probabilities[0, self.positive_class_index].detach().cpu()
            )
        self.results[key] = record
        self.predictions.append(outputs.detach().cpu())
        self.labels.append(target.detach().cpu())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward_batch(batch)
        probabilities = logits.softmax(dim=-1)
        if self.num_classes == 2:
            return probabilities[:, self.positive_class_index]
        return probabilities


class ViTRegressionModule(ViTClsRegModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.MSELoss()
        self.task_type = "regression"

    def prepare_target(self, target: torch.Tensor) -> torch.Tensor:
        # Always retain [B, output_dim], including B=1 and output_dim=1.
        return target.float().reshape(-1, self.num_outputs)

    def configure_metrics(self, prefix: str) -> MetricCollection:
        return MetricCollection(
            {
                f"{prefix}/MSE": MeanSquaredError(
                    num_outputs=self.num_outputs,
                ),
                f"{prefix}/MAE": MeanAbsoluteError(
                    num_outputs=self.num_outputs,
                ),
            }
        )

    def configure_test_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "MSE": MeanSquaredError(num_outputs=self.num_outputs),
                "MAE": MeanAbsoluteError(num_outputs=self.num_outputs),
            }
        )

    def _record_test_batch(self, outputs, target, batch) -> None:
        key = self._file_key(batch)
        prediction_values = outputs[0].detach().cpu().tolist()
        target_values = target[0].detach().cpu().tolist()

        self.results[key] = {
            "prediction": (
                prediction_values[0]
                if self.num_outputs == 1
                else prediction_values
            ),
            "label": (
                target_values[0]
                if self.num_outputs == 1
                else target_values
            ),
        }
        self.predictions.append(outputs.detach().cpu())
        self.labels.append(target.detach().cpu())
