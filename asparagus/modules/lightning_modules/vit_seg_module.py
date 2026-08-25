from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from asparagus.modules.lightning_modules.vit_task_base_module import (
    ViTTaskBaseModule,
)
from asparagus.modules.networks.vit_task_model import ViTSegModel
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall


class DiceBCELoss(nn.Module):
    """Binary BCE-with-logits plus soft Dice over valid (unpadded) voxels."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        positive_weight: Optional[float] = None,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero.")
        if positive_weight is not None and positive_weight <= 0:
            raise ValueError("positive_weight must be positive.")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        value = None if positive_weight is None else torch.tensor([positive_weight])
        self.register_buffer("positive_weight", value, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)
        mask = valid_mask.to(dtype=logits.dtype)
        pos_weight = (
            None
            if self.positive_weight is None
            else self.positive_weight.to(device=logits.device, dtype=logits.dtype)
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="none"
        )
        bce = (bce * mask).sum() / mask.sum().clamp_min(1.0)

        probabilities = logits.sigmoid()
        reduce_dims = tuple(range(1, logits.ndim))
        intersection = (probabilities * target * mask).sum(dim=reduce_dims)
        denominator = ((probabilities + target) * mask).sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return self.bce_weight * bce + self.dice_weight * (1.0 - dice.mean())


class ViTSegmentationModule(ViTTaskBaseModule):
    """Lightning module for full-volume binary BrainDINO segmentation."""

    def __init__(
        self,
        *args,
        label_key: str = "SEG_label",
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        positive_weight: Optional[float] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.unwrap_compiled_model(), ViTSegModel):
            raise TypeError("ViTSegmentationModule requires a ViTSegModel.")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")
        self.label_key = label_key
        self.threshold = float(threshold)
        self.loss = DiceBCELoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            positive_weight=positive_weight,
        )
        self.train_metrics = self._make_metrics("train")
        self.val_metrics = self._make_metrics("val")
        self.test_metrics = self._make_metrics("test")

    def _make_metrics(self, prefix: str) -> MetricCollection:
        return MetricCollection(
            {
                f"{prefix}/dice": BinaryF1Score(threshold=self.threshold),
                f"{prefix}/precision": BinaryPrecision(threshold=self.threshold),
                f"{prefix}/sensitivity": BinaryRecall(threshold=self.threshold),
            }
        )

    def _prepare_target(self, batch: Mapping[str, Any], logits: torch.Tensor):
        if self.label_key in batch:
            target = batch[self.label_key]
        elif self.label_key == "SEG_label" and "label" in batch:
            target = batch["label"]
        else:
            raise KeyError(f"Batch does not contain segmentation key {self.label_key!r}.")
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        if target.shape != logits.shape:
            raise ValueError(
                f"Segmentation target {tuple(target.shape)} must match logits "
                f"{tuple(logits.shape)}."
            )
        if logits.shape[1] != 1:
            raise ValueError("This module supports one-channel binary logits only.")
        return target.to(device=logits.device, dtype=logits.dtype)

    @staticmethod
    def _valid_shapes(batch: Mapping[str, Any], logits: torch.Tensor):
        info = batch.get("info", {})
        shapes = batch.get("valid_spatial_shapes", info.get("valid_spatial_shapes"))
        if shapes is None:
            return torch.tensor(
                logits.shape[2:], device=logits.device, dtype=torch.long
            ).repeat(logits.shape[0], 1)
        shapes = torch.as_tensor(shapes, device=logits.device, dtype=torch.long)
        if shapes.ndim == 1:
            shapes = shapes.unsqueeze(0)
        if shapes.shape != (logits.shape[0], 3):
            raise ValueError("valid_spatial_shapes must have shape [B, 3].")
        maximum = torch.tensor(logits.shape[2:], device=logits.device)
        return torch.minimum(shapes, maximum)

    @staticmethod
    def _valid_mask(logits: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        h = torch.arange(logits.shape[2], device=logits.device)[None, :, None, None]
        w = torch.arange(logits.shape[3], device=logits.device)[None, None, :, None]
        d = torch.arange(logits.shape[4], device=logits.device)[None, None, None, :]
        mask = (
            (h < shapes[:, 0, None, None, None])
            & (w < shapes[:, 1, None, None, None])
            & (d < shapes[:, 2, None, None, None])
        )
        return mask.unsqueeze(1)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        logits = self.forward_batch(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("The segmentation model produced non-finite logits.")
        target = self._prepare_target(batch, logits)
        shapes = self._valid_shapes(batch, logits)
        valid_mask = self._valid_mask(logits, shapes)
        loss = self.loss(logits, target, valid_mask)
        self.log(
            f"{stage}/loss", loss, on_step=False, on_epoch=True,
            sync_dist=True, batch_size=logits.shape[0],
        )
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(logits[valid_mask], target[valid_mask].long())
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def _finish_metrics(self, metrics: MetricCollection) -> None:
        self.log_dict(metrics.compute(), on_epoch=True, sync_dist=False)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._finish_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._finish_metrics(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx):
        logits = self.forward_batch(batch)
        target = self._prepare_target(batch, logits)
        shapes = self._valid_shapes(batch, logits)
        valid_mask = self._valid_mask(logits, shapes)
        self.test_metrics.update(logits[valid_mask], target[valid_mask].long())
        return logits.sigmoid()

    def on_test_epoch_end(self) -> None:
        self._finish_metrics(self.test_metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward_batch(batch).sigmoid()
