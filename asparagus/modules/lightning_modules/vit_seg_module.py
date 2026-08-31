from __future__ import annotations
import torch.distributed as dist
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from asparagus.modules.lightning_modules.vit_task_base_module import (
    ViTTaskBaseModule,
)
from asparagus.modules.networks.vit_task_model import ViTSegModel
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np
from lightning.pytorch.loggers import WandbLogger


class TinyForegroundFocalTverskyLoss(nn.Module):
    """Loss for mutually exclusive segmentation with a tiny foreground.

    The focal cross-entropy is reduced separately for background and for each
    foreground class present in a subject. Millions of background voxels
    therefore cannot dilute the foreground gradient. The Tversky component
    weights false negatives more strongly than false positives and directly
    opposes an all-background solution.

    All probability calculations run in float32 even under mixed precision to
    prevent small foreground probabilities from underflowing.
    """

    def __init__(
        self,
        num_classes: int,
        focal_weight: float = 0.35,
        tversky_weight: float = 0.65,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.20,
        tversky_beta: float = 0.80,
        tversky_gamma: float = 1.0,
        background_focal_weight: float = 0.25,
        absent_foreground_weight: float = 0.10,
        class_weights: Optional[Sequence[float]] = None,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and be >= 2.")
        if focal_weight < 0.0 or tversky_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if focal_weight + tversky_weight <= 0.0:
            raise ValueError("At least one loss weight must be positive.")
        if focal_gamma < 0.0 or tversky_gamma <= 0.0:
            raise ValueError("Invalid focal exponent.")
        if tversky_alpha < 0.0 or tversky_beta < 0.0:
            raise ValueError("Tversky alpha and beta must be non-negative.")
        if tversky_alpha + tversky_beta <= 0.0:
            raise ValueError("Tversky alpha + beta must be positive.")
        if background_focal_weight <= 0.0:
            raise ValueError("background_focal_weight must be positive.")
        if absent_foreground_weight < 0.0:
            raise ValueError("absent_foreground_weight cannot be negative.")
        if smooth <= 0.0:
            raise ValueError("smooth must be positive.")

        total = float(focal_weight + tversky_weight)
        self.num_classes = int(num_classes)
        self.focal_weight = float(focal_weight) / total
        self.tversky_weight = float(tversky_weight) / total
        self.focal_gamma = float(focal_gamma)
        tversky_total = float(tversky_alpha + tversky_beta)
        self.tversky_alpha = float(tversky_alpha) / tversky_total
        self.tversky_beta = float(tversky_beta) / tversky_total
        self.tversky_gamma = float(tversky_gamma)
        self.background_focal_weight = float(background_focal_weight)
        self.absent_foreground_weight = float(absent_foreground_weight)
        self.smooth = float(smooth)

        if class_weights is None:
            weights = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            if len(class_weights) != self.num_classes:
                raise ValueError(
                    "class_weights must contain one value per output class."
                )
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError(
                    "Every class weight must be finite and positive."
                )
        # Persistent=False preserves compatibility with older downstream
        # checkpoints that did not contain loss buffers.
        self.register_buffer("class_weights", weights, persistent=False)

    def _validate_inputs(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if logits.ndim != 5 or logits.shape[1] != self.num_classes:
            raise ValueError(
                "logits must be [B,C,H,W,D] with "
                f"C={self.num_classes}; got {tuple(logits.shape)}."
            )
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"target must be [B,H,W,D], got {tuple(target.shape)}."
            )
        if valid_mask.shape != target.shape:
            raise ValueError("valid_mask must have the target shape.")

    def _balanced_focal_loss(
        self,
        log_probabilities: torch.Tensor,
        probabilities: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target_log_probability = log_probabilities.gather(
            0,
            target.unsqueeze(0),
        ).squeeze(0)
        target_probability = probabilities.gather(
            0,
            target.unsqueeze(0),
        ).squeeze(0)
        voxel_focal = -(
            (1.0 - target_probability).pow(self.focal_gamma)
            * target_log_probability
        )

        group_losses = []
        group_weights = []
        for class_id in range(self.num_classes):
            class_mask = target == class_id
            if not class_mask.any():
                continue
            group_losses.append(voxel_focal[class_mask].mean())
            weight = self.class_weights[class_id]
            if class_id == 0:
                weight = weight * self.background_focal_weight
            group_weights.append(weight)

        if not group_losses:
            raise RuntimeError("Subject contains no valid target voxels.")
        losses = torch.stack(group_losses)
        weights = torch.stack(group_weights).to(losses)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def _foreground_tversky_loss(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        for class_id in range(1, self.num_classes):
            predicted = probabilities[class_id]
            expected = (target == class_id).to(predicted.dtype)
            foreground_count = expected.sum()

            if foreground_count > 0:
                true_positive = (predicted * expected).sum()
                false_positive = (predicted * (1.0 - expected)).sum()
                false_negative = ((1.0 - predicted) * expected).sum()
                score = (
                    true_positive + self.smooth
                ) / (
                    true_positive
                    + self.tversky_alpha * false_positive
                    + self.tversky_beta * false_negative
                    + self.smooth
                )
                terms.append((1.0 - score).pow(self.tversky_gamma))
            elif self.absent_foreground_weight > 0.0:
                # Negative subjects still suppress false positives, but this
                # term is deliberately weaker than a positive-subject term.
                terms.append(
                    self.absent_foreground_weight * predicted.mean()
                )

        if not terms:
            return probabilities.sum() * 0.0
        return torch.stack(terms).mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_components: bool = False,
    ):
        self._validate_inputs(logits, target, valid_mask)

        subject_losses = []
        focal_losses = []
        tversky_losses = []
        for index in range(logits.shape[0]):
            valid = valid_mask[index]
            if not valid.any():
                raise RuntimeError(f"Subject {index} contains no valid voxels.")

            # Flatten only valid voxels; padded background never enters either
            # component. Float32 protects tiny softmax values under bf16 AMP.
            subject_logits = logits[index, :, valid].float()
            subject_target = target[index, valid]
            log_probabilities = F.log_softmax(subject_logits, dim=0)
            probabilities = log_probabilities.exp()

            focal = self._balanced_focal_loss(
                log_probabilities,
                probabilities,
                subject_target,
            )
            tversky = self._foreground_tversky_loss(
                probabilities,
                subject_target,
            )
            subject_losses.append(
                self.focal_weight * focal
                + self.tversky_weight * tversky
            )
            focal_losses.append(focal)
            tversky_losses.append(tversky)

        loss = torch.stack(subject_losses).mean()
        if not return_components:
            return loss
        return loss, {
            "balanced_focal": torch.stack(focal_losses).mean().detach(),
            "foreground_tversky": (
                torch.stack(tversky_losses).mean().detach()
            ),
        }


class ViTSegmentationModule(ViTTaskBaseModule):
    """Padding-aware, mutually-exclusive multiclass ViT segmentation."""

    def __init__(
        self,
        *args,
        num_classes: int,
        label_key: str = "SEG_label",
        ce_weight: float = 0.3,
        dice_weight: float = 0.7,
        class_weights: Optional[Sequence[float]] = None,
        include_background_in_dice: bool = False,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.20,
        tversky_beta: float = 0.80,
        tversky_gamma: float = 1.0,
        background_focal_weight: float = 0.25,
        absent_foreground_weight: float = 0.10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        model = self.unwrap_compiled_model()
        if not isinstance(model, ViTSegModel):
            raise TypeError("ViTSegmentationModule requires ViTSegModel.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if int(model.output_channels) != int(num_classes):
            raise ValueError(
                "Model output_channels and num_classes disagree: "
                f"{model.output_channels} != {num_classes}."
            )

        self.num_classes = int(num_classes)
        self.label_key = str(label_key)
        if include_background_in_dice:
            raise ValueError(
                "Background must not be included in the Tversky overlap "
                "for tiny-foreground segmentation."
            )
        self.loss = TinyForegroundFocalTverskyLoss(
            num_classes=self.num_classes,
            focal_weight=ce_weight,
            tversky_weight=dice_weight,
            focal_gamma=focal_gamma,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
            background_focal_weight=background_focal_weight,
            absent_foreground_weight=absent_foreground_weight,
            class_weights=class_weights,
        )
        self.train_confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )
        self.val_confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )
        self.test_confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )

    def _prepare_target(
        self,
        batch: Mapping[str, Any],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.label_key in batch:
            target = batch[self.label_key]
        elif self.label_key == "SEG_label" and "label" in batch:
            target = batch["label"]
        else:
            raise KeyError(
                f"Batch does not contain segmentation key {self.label_key!r}."
            )

        target = torch.as_tensor(target, device=logits.device)
        if target.ndim == 5 and target.shape[1] == 1:
            target = target[:, 0]
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "Target must be [B, H, W, D] or [B, 1, H, W, D]; "
                f"target={tuple(target.shape)}, logits={tuple(logits.shape)}."
            )
        if not torch.isfinite(target.float()).all():
            raise FloatingPointError("Segmentation target contains non-finite values.")
        rounded = target.round()
        if not torch.allclose(target.float(), rounded.float(), atol=1e-4, rtol=0.0):
            raise ValueError("Segmentation labels must be integer class IDs.")
        target = rounded.long()
        if (target < 0).any() or (target >= self.num_classes).any():
            minimum = int(target.min().detach().cpu())
            maximum = int(target.max().detach().cpu())
            raise ValueError(
                f"Target IDs must be in [0, {self.num_classes - 1}], "
                f"but observed [{minimum}, {maximum}]."
            )
        return target

    @staticmethod
    def _valid_shapes(
        batch: Mapping[str, Any],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        info = batch.get("info", {})
        shapes = None
        if isinstance(info, Mapping):
            shapes = info.get("valid_spatial_shapes")
        if shapes is None:
            shapes = batch.get("valid_spatial_shapes")
        if shapes is None:
            return torch.tensor(
                logits.shape[2:], device=logits.device, dtype=torch.long
            ).repeat(logits.shape[0], 1)

        shapes = torch.as_tensor(shapes, device=logits.device, dtype=torch.long)
        if shapes.ndim == 1:
            shapes = shapes.unsqueeze(0)
        if shapes.shape != (logits.shape[0], 3):
            raise ValueError(
                "valid_spatial_shapes must be [B, 3], got "
                f"{tuple(shapes.shape)}."
            )
        maximum = torch.tensor(
            logits.shape[2:], device=logits.device, dtype=torch.long
        )
        shapes = torch.minimum(shapes, maximum)
        if (shapes <= 0).any():
            raise ValueError("Every valid spatial dimension must be positive.")
        return shapes

    @staticmethod
    def _valid_mask(logits: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        height = torch.arange(logits.shape[2], device=logits.device)
        width = torch.arange(logits.shape[3], device=logits.device)
        depth = torch.arange(logits.shape[4], device=logits.device)
        return (
            (height[None, :, None, None] < shapes[:, 0, None, None, None])
            & (width[None, None, :, None] < shapes[:, 1, None, None, None])
            & (depth[None, None, None, :] < shapes[:, 2, None, None, None])
        )

    @staticmethod
    def _create_colored_segmentation_overlay(
            image: torch.Tensor,
            target: torch.Tensor,
            prediction: torch.Tensor,
            alpha: float = 0.55,
    ) -> tuple[np.ndarray, int]:
        """Create target/prediction RGB overlay from one 3-D modality.

        Colors:
            green  = target only
            red    = prediction only
            yellow = target/prediction overlap
        """
        image = image.detach().float().cpu()
        target = target.detach().bool().cpu()
        prediction = prediction.detach().bool().cpu()

        if image.ndim != 3:
            raise ValueError(
                f"image must be [H, W, D], got {tuple(image.shape)}."
            )

        if target.shape != image.shape or prediction.shape != image.shape:
            raise ValueError(
                "Image, target, and prediction must have matching shapes."
            )

        # Prefer a slice containing target. Prediction is used as a secondary
        # criterion when the target is empty.
        slice_scores = (
                2 * target.sum(dim=(0, 1))
                + prediction.sum(dim=(0, 1))
        )
        slice_index = int(slice_scores.argmax())

        image_slice = image[:, :, slice_index]
        target_slice = target[:, :, slice_index]
        prediction_slice = prediction[:, :, slice_index]

        # Robust grayscale normalization.
        finite_values = image_slice[torch.isfinite(image_slice)]
        if finite_values.numel() == 0:
            normalized = torch.zeros_like(image_slice)
        else:
            lower = torch.quantile(finite_values, 0.01)
            upper = torch.quantile(finite_values, 0.99)

            if float(upper - lower) < 1e-6:
                normalized = torch.zeros_like(image_slice)
            else:
                normalized = (
                        (image_slice - lower) / (upper - lower)
                ).clamp(0.0, 1.0)

        rgb = normalized.unsqueeze(-1).repeat(1, 1, 3)

        target_only = target_slice & ~prediction_slice
        prediction_only = prediction_slice & ~target_slice
        overlap = target_slice & prediction_slice

        green = torch.tensor([0.0, 1.0, 0.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        yellow = torch.tensor([1.0, 1.0, 0.0])

        for mask, color in (
                (target_only, green),
                (prediction_only, red),
                (overlap, yellow),
        ):
            if mask.any():
                rgb[mask] = (
                        (1.0 - alpha) * rgb[mask]
                        + alpha * color
                )

        rgb = (rgb.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
        return rgb, slice_index

    def _log_visualization(
            self,
            batch,
            target,
            logits,
            shapes,
            stage: str,
    ) -> None:
        height, width, depth = (
            int(value)
            for value in shapes[0].detach().cpu().tolist()
        )

        image = batch["image"][
            0,
            :,
            :height,
            :width,
            :depth,
        ]

        target_foreground = (
                target[
                    0,
                    :height,
                    :width,
                    :depth,
                ]
                > 0
        )

        prediction_foreground = (
                logits[
                    0,
                    :,
                    :height,
                    :width,
                    :depth,
                ].argmax(dim=0)
                > 0
        )

        info = batch.get("info", {})

        channel_mask = (
            info.get("channel_mask")
            if isinstance(info, Mapping)
            else None
        )
        modality = (
            info.get("modality")
            if isinstance(info, Mapping)
            else None
        )

        if channel_mask is None:
            active_channels = list(range(image.shape[0]))
        else:
            first_subject_mask = (
                torch.as_tensor(channel_mask)[0]
                .bool()
                .detach()
                .cpu()
            )
            active_channels = (
                first_subject_mask
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )

        file_paths = batch.get("file_path", [])
        if isinstance(file_paths, (list, tuple)):
            file_path = str(file_paths[0])
        else:
            file_path = str(file_paths)

        loggers = getattr(self.trainer, "loggers", None)
        if loggers is None:
            loggers = [self.logger]

        for channel_index in active_channels:
            modality_id = channel_index

            if modality is not None:
                modality_id = int(
                    torch.as_tensor(modality)[0, channel_index]
                )

            overlay, slice_index = (
                self._create_colored_segmentation_overlay(
                    image=image[channel_index],
                    target=target_foreground,
                    prediction=prediction_foreground,
                )
            )

            caption = (
                f"{file_path}\n"
                f"channel={channel_index}, "
                f"modality_id={modality_id}, "
                f"slice={slice_index}\n"
                "green=target only, red=prediction only, "
                "yellow=overlap"
            )

            key = (
                f"Media/{stage}/"
                f"modality_{channel_index}_id_{modality_id}"
            )

            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key=key,
                        images=[overlay],
                        caption=[caption],
                    )

    def _shared_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        logits = self.forward_batch(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Model produced non-finite logits.")
        target = self._prepare_target(batch, logits)
        shapes = self._valid_shapes(batch, logits)
        valid_mask = self._valid_mask(logits, shapes)
        loss, loss_components = self.loss(
            logits,
            target,
            valid_mask,
            return_components=True,
        )
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/balanced_focal_loss": (
                    loss_components["balanced_focal"]
                ),
                f"{stage}/foreground_tversky_loss": (
                    loss_components["foreground_tversky"]
                ),
            },
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=logits.shape[0],
        )

        confusion = (
            self.train_confusion if stage == "train" else self.val_confusion
        )
        predictions = logits.argmax(dim=1)
        confusion.update(predictions[valid_mask], target[valid_mask])

        probabilities = logits.softmax(dim=1)
        foreground_probability = probabilities[:, 1:].sum(dim=1)
        target_foreground = (target > 0) & valid_mask
        predicted_foreground = (predictions > 0) & valid_mask
        intersection = (
            foreground_probability * target_foreground.to(probabilities.dtype)
        )[valid_mask].sum()
        denominator = foreground_probability[valid_mask].sum() + (
            target_foreground[valid_mask].sum()
        )
        soft_dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)

        foreground_values = foreground_probability[target_foreground]
        background_values = foreground_probability[
            valid_mask & ~target_foreground
        ]
        nan = foreground_probability.new_tensor(float("nan"))
        mean_target_foreground_probability = (
            foreground_values.mean()
            if foreground_values.numel() > 0
            else nan
        )
        mean_background_foreground_probability = (
            background_values.mean()
            if background_values.numel() > 0
            else nan
        )
        self.log_dict(
            {
                f"{stage}/soft_dice": soft_dice,
                f"{stage}/target_foreground_voxels": target_foreground.sum().float(),
                f"{stage}/predicted_foreground_voxels": predicted_foreground.sum().float(),
                f"{stage}/correct_foreground_voxels": (
                    target_foreground & predicted_foreground
                ).sum().float(),
                f"{stage}/mean_foreground_probability": (
                    foreground_probability[valid_mask].mean()
                ),
                f"{stage}/mean_target_foreground_probability": (
                    mean_target_foreground_probability
                ),
                f"{stage}/mean_background_foreground_probability": (
                    mean_background_foreground_probability
                ),
            },
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=logits.shape[0],
        )

        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch > 0
            and self.log_image_every_n_epochs > 0
            and self.current_epoch % self.log_image_every_n_epochs == 0
            and self.logger is not None
        ):
            self._log_visualization(batch, target, logits, shapes, stage)
        return loss

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor):
        nan = torch.full_like(numerator, float("nan"))
        return torch.where(denominator > 0, numerator / denominator, nan)

    def _finish_metrics(
            self,
            confusion_metric: MulticlassConfusionMatrix,
            prefix: str,
    ) -> None:
        # Read the local fixed-size confusion matrix without triggering
        # TorchMetrics' implicit distributed synchronization.
        confusion = confusion_metric.confmat.detach().clone()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                confusion,
                op=dist.ReduceOp.SUM,
            )

        confusion = confusion.float()

        true_positive = confusion.diag()
        false_negative = confusion.sum(dim=1) - true_positive
        false_positive = confusion.sum(dim=0) - true_positive

        dice = self._safe_ratio(
            2.0 * true_positive,
            2.0 * true_positive
            + false_positive
            + false_negative,
        )
        precision = self._safe_ratio(
            true_positive,
            true_positive + false_positive,
        )
        sensitivity = self._safe_ratio(
            true_positive,
            true_positive + false_negative,
        )

        foreground = slice(1, None)
        values = {
            f"{prefix}/dice": torch.nanmean(dice[foreground]),
            f"{prefix}/precision": torch.nanmean(
                precision[foreground]
            ),
            f"{prefix}/sensitivity": torch.nanmean(
                sensitivity[foreground]
            ),
        }

        for class_id in range(1, self.num_classes):
            values[f"{prefix}/class_{class_id}_dice"] = (
                dice[class_id]
            )
            values[f"{prefix}/class_{class_id}_precision"] = (
                precision[class_id]
            )
            values[f"{prefix}/class_{class_id}_sensitivity"] = (
                sensitivity[class_id]
            )

        # Every rank has the same globally reduced values now, so do not
        # ask Lightning to synchronize them again.
        self.log_dict(
            values,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        confusion_metric.reset()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def on_train_epoch_end(self) -> None:
        self._finish_metrics(self.train_confusion, "train")

    def on_validation_epoch_end(self) -> None:
        self._finish_metrics(self.val_confusion, "val")

    def on_test_epoch_start(self) -> None:
        self.test_confusion.reset()

    def _masked_probabilities(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = logits.softmax(dim=1)
        probabilities = probabilities * valid_mask.unsqueeze(1).to(probabilities)
        probabilities[:, 0] += (~valid_mask).to(probabilities.dtype)
        return probabilities

    def test_step(self, batch, batch_idx):
        logits = self.forward_batch(batch)
        target = self._prepare_target(batch, logits)
        shapes = self._valid_shapes(batch, logits)
        valid_mask = self._valid_mask(logits, shapes)
        predictions = logits.argmax(dim=1)
        self.test_confusion.update(
            predictions[valid_mask], target[valid_mask]
        )
        return self._masked_probabilities(logits, valid_mask)

    def on_test_epoch_end(self) -> None:
        self._finish_metrics(self.test_confusion, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward_batch(batch)
        shapes = self._valid_shapes(batch, logits)
        valid_mask = self._valid_mask(logits, shapes)
        return self._masked_probabilities(logits, valid_mask)
