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
    BinaryAUROC,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)
import math
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)


class ViTLinearProbModule(ViTTaskBaseModule):
    """Inference-only Lightning module that exports raw CLS embeddings."""

    def training_step(self, batch, batch_idx):
        raise RuntimeError("ViTLinearProbModule is inference-only.")

    def validation_step(self, batch, batch_idx):
        raise RuntimeError("ViTLinearProbModule is inference-only.")

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        embedding = self.forward_batch(batch)
        if embedding.ndim != 2:
            raise RuntimeError(
                "ViTLinearProbModel must return [B, E], got "
                f"{tuple(embedding.shape)}."
            )
        return embedding.float()


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

        # Retain this name for compatibility with existing configurations.
        self.num_classes = self.num_outputs
        self.ignore_index_in_metrics = -1
        self.task_type = ""
        self.loss: nn.Module

        self.train_metrics: MetricCollection
        self.val_metrics: MetricCollection
        self.test_metrics: MetricCollection

        self.results: dict[str, Any] = {}
        self.predictions: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def _initialize_metrics(self) -> None:
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_test_metrics()

    @staticmethod
    def _infer_num_outputs(model: nn.Module) -> int:
        if hasattr(model, "num_classes"):
            return int(model.num_classes)

        if hasattr(model, "output_dim"):
            return int(model.output_dim)

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

    def prepare_metric_inputs(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return outputs, target

    def _shared_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        target = self.prepare_target(batch["CLSREG_label"])
        outputs = self.forward_batch(batch)

        self._validate_output_and_target(outputs, target)
        loss = self.compute_loss(outputs, target, stage)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=outputs.shape[0],
        )

        metrics = (
            self.train_metrics
            if stage == "train"
            else self.val_metrics
        )

        metric_outputs, metric_target = self.prepare_metric_inputs(
            outputs.detach(),
            target.detach(),
        )
        metrics.update(metric_outputs, metric_target)

        if (
            self.current_epoch > 0
            and batch_idx == 0
            and self.log_image_every_n_epochs > 0
            and self.current_epoch % self.log_image_every_n_epochs == 0
            and self.logger is not None
        ):
            visualization_batch = self._prepare_visualization_batch(
                batch=batch,
                target=target,
                outputs=outputs,
            )

            self._log_dict_of_images_to_wandb(
                visualization_batch,
                log_key=stage,
                task_type=self.task_type,
            )

        return loss

    def compute_loss(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        return self.loss(outputs, target)

    def _validate_output_and_target(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if outputs.ndim != 2:
            raise ValueError(
                "The task model must return [B, num_outputs], got "
                f"{tuple(outputs.shape)}."
            )

        if outputs.shape[1] != self.num_outputs:
            raise ValueError(
                f"Expected {self.num_outputs} outputs, "
                f"got {outputs.shape[1]}."
            )

        if self.task_type == "regression" and target.ndim != 2:
            raise ValueError(
                "Regression targets must be [B, output_dim], got "
                f"{tuple(target.shape)}."
            )

        if target.shape[0] != outputs.shape[0]:
            raise ValueError(
                f"Output batch size {outputs.shape[0]} does not match "
                f"target batch size {target.shape[0]}."
            )

        if not torch.isfinite(outputs).all():
            raise FloatingPointError(
                "The model produced non-finite outputs."
            )

    @staticmethod
    def _prepare_visualization_batch(
            batch: Mapping[str, Any],
            target: torch.Tensor,
            outputs: torch.Tensor,
    ) -> dict[str, Any]:
        """Prepare one unpadded subject for image logging.

        Full-volume batches contain high-end padding so subjects with different
        spatial dimensions can be stacked. Logging the padded batch directly can
        select slices from padding and make the anatomy appear displaced,
        truncated, or empty.
        """
        image = batch["image"]

        if image.ndim != 5:
            raise ValueError(
                "Expected batch image with shape [B, C, H, W, D], "
                f"received {tuple(image.shape)}."
            )

        info = batch.get("info", {})
        valid_shapes = info.get("valid_spatial_shapes")

        if valid_shapes is None:
            valid_shape = image.shape[-3:]
        else:
            valid_shapes = torch.as_tensor(
                valid_shapes,
                device=image.device,
                dtype=torch.long,
            )

            if valid_shapes.ndim == 1:
                valid_shapes = valid_shapes.unsqueeze(0)

            if valid_shapes.shape != (image.shape[0], 3):
                raise ValueError(
                    "valid_spatial_shapes must have shape [B, 3], "
                    f"received {tuple(valid_shapes.shape)}."
                )

            valid_shape = tuple(
                int(value)
                for value in valid_shapes[0].detach().cpu().tolist()
            )

        height, width, depth = valid_shape

        if (
                height < 1
                or width < 1
                or depth < 1
                or height > image.shape[2]
                or width > image.shape[3]
                or depth > image.shape[4]
        ):
            raise ValueError(
                f"Invalid visualization shape {valid_shape} for "
                f"batched image shape {tuple(image.shape)}."
            )

        # Log only one subject because different subjects can have different
        # unpadded dimensions and therefore cannot be stacked after cropping.
        visualization_image = image[
            0:1,
            :,
            :height,
            :width,
            :depth,
        ]

        file_paths = batch.get("file_path", [])
        if isinstance(file_paths, (list, tuple)):
            visualization_files = list(file_paths[:1])
        else:
            visualization_files = [file_paths]

        return {
            "input": (
                visualization_image
                .detach()
                .float()
                .cpu()
                .numpy()
            ),
            "target": (
                target[0:1]
                .detach()
                .float()
                .cpu()
                .numpy()
            ),
            "output": (
                outputs[0:1]
                .detach()
                .float()
                .cpu()
                .numpy()
            ),
            "file": visualization_files,
        }

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> None:
        self._shared_step(batch, batch_idx, "val")

    def _finish_metric_epoch(
        self,
        metrics: MetricCollection,
    ) -> None:
        values = metrics.compute()

        formatted_values = format_multilabel_metrics(
            values,
            ignore_index=self.ignore_index_in_metrics,
        )

        self.log_dict(
            formatted_values,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

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

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        outputs = self.forward_batch(batch)
        target = self.prepare_target(batch["CLSREG_label"])

        self._validate_output_and_target(outputs, target)

        metric_outputs, metric_target = self.prepare_metric_inputs(
            outputs.detach(),
            target.detach(),
        )
        self.test_metrics.update(metric_outputs, metric_target)

        self._record_test_batch(outputs, target, batch)
        return outputs

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
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
            sum(
                prediction.shape[0]
                for prediction in self.predictions
            ),
            metric_values,
        )

        self.test_metrics.reset()


class ViTClassificationModule(ViTClsRegModule):
    """Multiclass classification trained with ordinary cross-entropy."""

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
                "ViTClassificationModule requires at least "
                "two output logits."
            )

        if not 0 <= positive_class_index < self.num_classes:
            raise ValueError(
                "positive_class_index is outside the class range."
            )

        self.positive_class_index = int(
            positive_class_index
        )

        if loss_weight is not None:
            loss_weight = torch.as_tensor(
                loss_weight,
                dtype=torch.float32,
            ).reshape(-1)

            if loss_weight.numel() != self.num_classes:
                raise ValueError(
                    "loss_weight must contain exactly "
                    f"{self.num_classes} values, but received "
                    f"{loss_weight.numel()}."
                )

        self.loss = nn.CrossEntropyLoss(
            weight=loss_weight,
            label_smoothing=label_smoothing,
        )

        # Training and validation use the same proper scoring rule. Dataset
        # imbalance is handled by sampling rather than by modifying the loss.
        self.validation_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        self.task_type = "classification"
        self._initialize_metrics()

    def prepare_target(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.reshape(-1).long()

        if target.numel() > 0:
            minimum = int(target.min())
            maximum = int(target.max())

            if minimum < 0 or maximum >= self.num_classes:
                raise ValueError(
                    "Classification targets must be in "
                    f"[0, {self.num_classes - 1}], got range "
                    f"[{minimum}, {maximum}]."
                )

        return target

    def compute_loss(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        if stage == "train":
            return self.loss(outputs, target)

        if stage == "val":
            return self.validation_loss(outputs, target)

        raise ValueError(
            f"Unsupported stage: {stage}."
        )

    def configure_metrics(self, prefix: str) -> MetricCollection:
        if self.num_classes == 2:
            return MetricCollection(
                {
                    f"{prefix}/acc": BinaryAccuracy(),
                    f"{prefix}/auroc": BinaryAUROC(),
                }
            )
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

    def configure_test_metrics(
        self,
    ) -> MetricCollection:
        return MetricCollection(
            {
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

    def prepare_metric_inputs(
            self,
            outputs: torch.Tensor,
            target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = outputs.softmax(dim=-1)
        if self.num_classes == 2:
            return probabilities[:, self.positive_class_index], target
        return probabilities, target

    def _record_test_batch(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> None:
        probabilities = outputs.softmax(dim=-1)
        prediction = outputs.argmax(dim=-1)

        file_paths = batch["file_path"]
        if not isinstance(file_paths, (list, tuple)):
            file_paths = [file_paths]

        for index, file_path in enumerate(file_paths):
            record = {
                "prediction": int(
                    prediction[index].detach().cpu()
                ),
                "probabilities": (
                    probabilities[index]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "label": int(
                    target[index].detach().cpu()
                ),
            }

            if self.num_classes == 2:
                record["probability"] = float(
                    probabilities[
                        index,
                        self.positive_class_index,
                    ]
                    .detach()
                    .cpu()
                )

            self.results[str(file_path)] = record

        self.predictions.append(
            outputs.detach().cpu()
        )
        self.labels.append(
            target.detach().cpu()
        )

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        logits = self.forward_batch(batch)
        probabilities = logits.softmax(dim=-1)

        if self.num_classes == 2:
            return probabilities[
                :,
                self.positive_class_index,
            ]

        return probabilities


class ViTRegressionModule(ViTClsRegModule):
    def __init__(
        self,
        *args,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not math.isfinite(target_mean):
            raise ValueError(
                f"target_mean must be finite, got {target_mean}."
            )

        if not math.isfinite(target_std) or target_std <= 0.0:
            raise ValueError(
                "target_std must be finite and positive, "
                f"got {target_std}."
            )

        self.register_buffer(
            "target_mean",
            torch.tensor(
                float(target_mean),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "target_std",
            torch.tensor(
                float(target_std),
                dtype=torch.float32,
            ),
        )

        # MSE is now calculated in standardized target space.
        self.loss = nn.MSELoss()
        self.task_type = "regression"
        self._initialize_metrics()

    def normalize_target(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            target - self.target_mean
        ) / self.target_std

    def denormalize_target(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            target * self.target_std
            + self.target_mean
        )

    def prepare_target(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float().reshape(
            -1,
            self.num_outputs,
        )
        return self.normalize_target(target)

    def configure_metrics(
            self,
            prefix: str,
    ) -> MetricCollection:
        return MetricCollection(
            {
                f"{prefix}/MAE": MeanAbsoluteError(),
                f"{prefix}/MSE": MeanSquaredError(),
                f"{prefix}/R2": R2Score(),
            }
        )

    def configure_test_metrics(
            self,
    ) -> MetricCollection:
        return MetricCollection(
            {
                "MAE": MeanAbsoluteError(),
                "MSE": MeanSquaredError(),
                "R2": R2Score(),
            }
        )

    def prepare_metric_inputs(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Metrics are reported in years.
        predictions_years = self.denormalize_target(outputs)
        targets_years = self.denormalize_target(target)

        return predictions_years, targets_years

    def _record_test_batch(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> None:
        outputs_years = self.denormalize_target(outputs)
        targets_years = self.denormalize_target(target)

        file_paths = batch["file_path"]
        if not isinstance(file_paths, (list, tuple)):
            file_paths = [file_paths]

        for index, file_path in enumerate(file_paths):
            prediction_values = (
                outputs_years[index]
                .detach()
                .cpu()
                .tolist()
            )
            target_values = (
                targets_years[index]
                .detach()
                .cpu()
                .tolist()
            )

            self.results[str(file_path)] = {
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

        self.predictions.append(
            outputs_years.detach().cpu()
        )
        self.labels.append(
            targets_years.detach().cpu()
        )

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        normalized_predictions = self.forward_batch(batch)
        return self.denormalize_target(
            normalized_predictions
        )


def configure_segmentation_metrics(
    prefix: str,
) -> MetricCollection:
    """Configure informative binary segmentation metrics.

    Dice/F1 is the primary overlap measure. Precision and sensitivity
    expose false-positive and false-negative behavior.

    Accuracy and specificity are omitted because they are dominated by
    background voxels. Voxel-level AUROC and average precision are omitted
    because they measure ranking rather than the final binary mask.
    """

    return MetricCollection(
        {
            f"{prefix}/dice": BinaryF1Score(),
            f"{prefix}/precision": BinaryPrecision(),
            f"{prefix}/sensitivity": BinaryRecall(),
        }
    )


def prepare_segmentation_metric_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare binary segmentation logits and targets for TorchMetrics."""

    if logits.ndim < 3 or logits.shape[1] != 1:
        raise ValueError(
            "Binary segmentation logits must have shape "
            "[B, 1, ...], got "
            f"{tuple(logits.shape)}."
        )

    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)

    if target.shape != logits.shape:
        raise ValueError(
            f"Target shape {tuple(target.shape)} does not match "
            f"logits shape {tuple(logits.shape)}."
        )

    metric_logits = logits[:, 0].reshape(-1)
    metric_target = target[:, 0].reshape(-1).long()

    return metric_logits, metric_target
