import copy
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import wandb
from lightning.pytorch.loggers import WandbLogger
from torchvision.utils import make_grid

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from asparagus.functional.metrics import (
    features as feat_metrics,
    performance as perf_metrics,
    stability as stability_metrics,
)
from asparagus.modules.lightning_modules.base_module import BaseModule
from asparagus.functional.lr_scheduling import simple_warmup_cosine_decay_schedule


ModelOutput = Dict[str, Any]
ViewInfo = Dict[str, Any]


class BrainDinoModule(BaseModule):
    """
    Abstract Lightning training module for BrainDINO pretraining.

    Objectives:
        1. DINO image-level self-distillation.
        2. iBOT masked patch-level self-distillation.
        3. Regional consistency over pooled 3D patch features.
        4. KoLeo regularization over global features.

    The supplied ``model`` owns all architectural details.

    Required model call:

        output = model(
            x,
            spacing=spacing,
            modality=modality,
            mask=mask,
        )

    Required model output:

        {
            "cls_features": Tensor[B, E],
            "patch_features": Tensor[B, N, E],
            "cls_projection": Tensor[B, K_cls],
            "patch_projection": Tensor[B, N, K_patch],
            "patch_grid_shape": (D_tokens, H_tokens, W_tokens),
        }

    Expected training batch:

        {
            "global_crops": [Tensor[B, C, D, H, W], ...],
            "local_crops": [Tensor[B, C, D, H, W], ...],
            "global_masks": [BoolTensor[B, N], ...],
            "global_info": [{"spacing": ..., "modality": ...}, ...],
            "local_info": [{"spacing": ..., "modality": ...}, ...],
            "transforms_applied": optional,
        }

    Subclasses implement the four objective methods and may override
    ``update_objective_state`` for centers or other running statistics.
    """

    REQUIRED_OUTPUT_KEYS = {
        "cls_features",
        "patch_features",
        "cls_projection",
        "patch_projection",
        "patch_grid_shape",
    }

    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-4,

            # Retained only because BaseModule expects it.
            warmup_epochs: int = 0,

            warmup_ratio: float = 0.10,
            cosine_period_ratio: float = 1.0,
            minimum_lr: float = 1e-6,

            compile_mode: Optional[str] = None,
            train_transforms: Optional[transforms.Compose] = None,
            test_transforms: Optional[transforms.Compose] = None,
            val_transforms: Optional[transforms.Compose] = None,
            optimizer: str = "AdamW",
            mlflow_logging: bool = False,
            log_every_n_steps: int = 50,
            weight_decay: float = 0.04,
            nesterov: bool = True,
            momentum: float = 0.9,
            weights: Optional[dict] = None,
            dino_loss_weight: float = 1.0,
            ibot_loss_weight: float = 1.0,
            region_loss_weight: float = 1.0,
            koleo_loss_weight: float = 0.1,
            teacher_momentum: float = 0.994,
            teacher_momentum_final: float = 1.0,
            teacher_temperature: float = 0.07,
            teacher_warmup_temperature: float = 0.04,
            student_temperature: float = 0.1,
            center_momentum: float = 0.9,
            region_pool_size: Tuple[int, int, int] = (2, 2, 2),
            region_max_masked_fraction: float = 0.5,
            koleo_eps: float = 1e-8,

            visual_log_every_n_steps: int = 10,
            visual_log_max_channels: int = 3,
            visual_log_sample_index: int = 0,
    ) -> None:
        super().__init__(
            model=model,
            warmup_epochs=warmup_epochs,
            learning_rate=learning_rate,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            weights=weights,
        )

        # BrainDINO uses a step-based scheduler controlled by trainer.max_steps.
        self.warmup_ratio = float(warmup_ratio)
        self.minimum_lr = float(minimum_lr)

        self.mlflow_logging = mlflow_logging
        self.log_every_n_steps = log_every_n_steps

        self.dino_loss_weight = float(dino_loss_weight)
        self.ibot_loss_weight = float(ibot_loss_weight)
        self.region_loss_weight = float(region_loss_weight)
        self.koleo_loss_weight = float(koleo_loss_weight)

        self.teacher_momentum = float(teacher_momentum)
        self.teacher_momentum_final = float(teacher_momentum_final)
        self.teacher_temperature = float(teacher_temperature)
        self.teacher_warmup_temperature = float(
            teacher_warmup_temperature
        )

        self.student_temperature = float(student_temperature)
        self.center_momentum = float(center_momentum)
        self.region_pool_size = tuple(
            int(value) for value in region_pool_size
        )
        self.region_max_masked_fraction = float(
            region_max_masked_fraction
        )
        self.koleo_eps = float(koleo_eps)

        self.visual_log_every_n_steps = int(visual_log_every_n_steps)
        self.visual_log_max_channels = int(visual_log_max_channels)
        self.visual_log_sample_index = int(visual_log_sample_index)

        # Prevent duplicate logging during gradient accumulation.
        self._last_visual_log_step = -1
        # W&B visual metric definitions are initialized lazily.
        self._wandb_visual_metrics_defined = False

        self._validate_hyperparameters()

        self.teacher = copy.deepcopy(
            self.unwrap_compiled_model()
        )
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.register_buffer(
            "dino_center",
            torch.empty(0),
            persistent=True,
        )
        self.register_buffer(
            "ibot_center",
            torch.empty(0),
            persistent=True,
        )

    def _validate_hyperparameters(self) -> None:
        for name, value in {
            "dino_loss_weight": self.dino_loss_weight,
            "ibot_loss_weight": self.ibot_loss_weight,
            "region_loss_weight": self.region_loss_weight,
            "koleo_loss_weight": self.koleo_loss_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        if not 0.0 <= self.teacher_momentum <= 1.0:
            raise ValueError("teacher_momentum must be between 0 and 1.")
        if not 0.0 <= self.teacher_momentum_final <= 1.0:
            raise ValueError(
                "teacher_momentum_final must be between 0 and 1."
            )
        if self.teacher_momentum_final < self.teacher_momentum:
            raise ValueError(
                "teacher_momentum_final must be >= teacher_momentum."
            )
        if self.teacher_temperature <= 0.0:
            raise ValueError("teacher_temperature must be positive.")
        if self.teacher_warmup_temperature <= 0.0:
            raise ValueError(
                "teacher_warmup_temperature must be positive."
            )

        if self.student_temperature <= 0.0:
            raise ValueError("student_temperature must be positive.")
        if not 0.0 <= self.center_momentum < 1.0:
            raise ValueError(
                "center_momentum must be in the interval [0, 1)."
            )
        if len(self.region_pool_size) != 3 or any(
            value <= 0 for value in self.region_pool_size
        ):
            raise ValueError(
                "region_pool_size must contain three positive integers."
            )
        if not 0.0 <= self.region_max_masked_fraction <= 1.0:
            raise ValueError(
                "region_max_masked_fraction must be between 0 and 1."
            )
        if self.koleo_eps <= 0.0:
            raise ValueError("koleo_eps must be positive.")

        if not 0.0 < self.warmup_ratio < 1.0:
            raise ValueError(
                "warmup_ratio must be strictly between 0 and 1."
            )

        if not 0.0 < self.cosine_period_ratio <= 1.0:
            raise ValueError(
                "cosine_period_ratio must be in the interval (0, 1]."
            )

        if self.minimum_lr < 0.0:
            raise ValueError("minimum_lr must be non-negative.")

        if self.minimum_lr >= self.learning_rate:
            raise ValueError(
                "minimum_lr must be smaller than learning_rate. "
                f"Received minimum_lr={self.minimum_lr} and "
                f"learning_rate={self.learning_rate}."
            )

        if self.visual_log_every_n_steps < 0:
            raise ValueError(
                "visual_log_every_n_steps must be non-negative."
            )

        if self.visual_log_max_channels <= 0:
            raise ValueError(
                "visual_log_max_channels must be positive."
            )

        if self.visual_log_sample_index < 0:
            raise ValueError(
                "visual_log_sample_index must be non-negative."
            )

    def _total_optimizer_steps(self) -> int:
        """
        Return the configured number of optimizer updates.

        Lightning's max_steps/global_step count optimizer updates, so gradient
        accumulation is already accounted for.
        """
        total_steps = self.trainer.max_steps

        if total_steps is None or total_steps <= 0:
            total_steps = self.trainer.estimated_stepping_batches

        if total_steps is None or total_steps <= 0:
            raise RuntimeError(
                "Could not determine the total number of optimizer steps."
            )

        return int(total_steps)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _ensure_centers(
        self,
        cls_projection: torch.Tensor,
        patch_projection: torch.Tensor,
    ) -> None:
        """Initialize and validate the running teacher centers."""
        cls_dim = cls_projection.shape[-1]
        patch_dim = patch_projection.shape[-1]

        if self.dino_center.numel() == 0:
            self.dino_center = torch.zeros(
                1,
                cls_dim,
                device=cls_projection.device,
                dtype=torch.float32,
            )
        elif self.dino_center.shape != (1, cls_dim):
            raise ValueError(
                "The CLS projection dimension changed during training: "
                f"center={tuple(self.dino_center.shape)}, "
                f"projection={tuple(cls_projection.shape)}."
            )

        if self.ibot_center.numel() == 0:
            self.ibot_center = torch.zeros(
                1,
                patch_dim,
                device=patch_projection.device,
                dtype=torch.float32,
            )
        elif self.ibot_center.shape != (1, patch_dim):
            raise ValueError(
                "The patch projection dimension changed during training: "
                f"center={tuple(self.ibot_center.shape)}, "
                f"projection={tuple(patch_projection.shape)}."
            )

    @staticmethod
    def _teacher_cross_entropy(
        student_projection: torch.Tensor,
        teacher_probabilities: torch.Tensor,
        student_temperature: float,
    ) -> torch.Tensor:
        """Cross-entropy with a stop-gradient teacher distribution."""
        student_log_probabilities = F.log_softmax(
            student_projection.float() / student_temperature,
            dim=-1,
        )
        return -(
            teacher_probabilities.float()
            * student_log_probabilities
        ).sum(dim=-1)

    def compute_dino_loss(
        self,
        student_global: Sequence[ModelOutput],
        student_local: Sequence[ModelOutput],
        teacher_global: Sequence[ModelOutput],
        teacher_temperature: float,
    ) -> torch.Tensor:
        """
        Image-level DINO self-distillation.

        Every teacher global view supervises all student views except the
        student global view generated from the same crop.
        """
        if not teacher_global:
            raise ValueError(
                "At least one teacher global view is required."
            )

        self._ensure_centers(
            teacher_global[0]["cls_projection"],
            teacher_global[0]["patch_projection"],
        )

        with torch.no_grad():
            teacher_probabilities = [
                F.softmax(
                    (
                        output["cls_projection"].float()
                        - self.dino_center
                    )
                    / teacher_temperature,
                    dim=-1,
                ).detach()
                for output in teacher_global
            ]

        student_views = list(student_global) + list(student_local)
        num_student_global = len(student_global)
        loss_terms: List[torch.Tensor] = []

        for teacher_index, teacher_probability in enumerate(
            teacher_probabilities
        ):
            for student_index, student_output in enumerate(student_views):
                if (
                    student_index < num_student_global
                    and student_index == teacher_index
                ):
                    continue

                loss_terms.append(
                    self._teacher_cross_entropy(
                        student_output["cls_projection"],
                        teacher_probability,
                        self.student_temperature,
                    ).mean()
                )

        if not loss_terms:
            raise RuntimeError(
                "No valid teacher-student view pairs were available "
                "for the DINO loss."
            )

        return torch.stack(loss_terms).mean()

    def compute_ibot_loss(
        self,
        student_global: Sequence[ModelOutput],
        teacher_global: Sequence[ModelOutput],
        global_masks: Sequence[torch.Tensor],
        teacher_temperature: float,
    ) -> torch.Tensor:
        """
        Masked patch-level iBOT self-distillation.

        Only student patch positions marked ``True`` in ``global_masks`` are
        optimized. The matching unmasked teacher patch provides the target.
        """
        if not (
            len(student_global)
            == len(teacher_global)
            == len(global_masks)
        ):
            raise ValueError(
                "student_global, teacher_global, and global_masks must "
                "have equal lengths."
            )

        if not teacher_global:
            raise ValueError(
                "At least one teacher global view is required."
            )

        self._ensure_centers(
            teacher_global[0]["cls_projection"],
            teacher_global[0]["patch_projection"],
        )

        loss_terms: List[torch.Tensor] = []

        for student_output, teacher_output, mask in zip(
            student_global,
            teacher_global,
            global_masks,
        ):
            student_projection = student_output["patch_projection"]
            teacher_projection = teacher_output["patch_projection"]

            if mask.shape != student_projection.shape[:2]:
                raise ValueError(
                    "Each global mask must match the corresponding patch "
                    "projection shape [B, N]. "
                    f"mask={tuple(mask.shape)}, "
                    f"projection={tuple(student_projection.shape)}."
                )

            if teacher_projection.shape[:2] != student_projection.shape[:2]:
                raise ValueError(
                    "Matching student and teacher global views must contain "
                    "the same patch layout."
                )

            if not mask.any():
                continue

            with torch.no_grad():
                teacher_probability = F.softmax(
                    (
                        teacher_projection.float()
                        - self.ibot_center
                    )
                    / teacher_temperature,
                    dim=-1,
                ).detach()

            loss_terms.append(
                self._teacher_cross_entropy(
                    student_projection[mask],
                    teacher_probability[mask],
                    self.student_temperature,
                ).mean()
            )

        if not loss_terms:
            # This keeps the graph/device/dtype valid while allowing batches
            # with no masked tokens.
            return student_global[0]["patch_projection"].sum() * 0.0

        return torch.stack(loss_terms).mean()

    @staticmethod
    def _patches_to_grid(
        patch_features: torch.Tensor,
        patch_grid_shape: Sequence[int],
    ) -> torch.Tensor:
        """Convert [B, N, E] patch features to [B, E, D, H, W]."""
        depth, height, width = map(int, patch_grid_shape)
        batch_size, num_patches, feature_dim = patch_features.shape

        if depth * height * width != num_patches:
            raise ValueError(
                "patch_grid_shape is incompatible with patch_features."
            )

        return (
            patch_features
            .reshape(batch_size, depth, height, width, feature_dim)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def compute_region_loss(
        self,
        student_global: Sequence[ModelOutput],
        teacher_global: Sequence[ModelOutput],
        global_masks: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Regional consistency over pooled 3D patch features.

        Patch features are reshaped into their 3D token grid and average-pooled
        into non-overlapping local regions. A region is used when its masked
        fraction does not exceed ``region_max_masked_fraction``.

        No extra regional head or reconstruction decoder is required.
        """
        if not (
            len(student_global)
            == len(teacher_global)
            == len(global_masks)
        ):
            raise ValueError(
                "student_global, teacher_global, and global_masks must "
                "have equal lengths."
            )

        loss_terms: List[torch.Tensor] = []

        for student_output, teacher_output, mask in zip(
            student_global,
            teacher_global,
            global_masks,
        ):
            student_grid_shape = tuple(
                student_output["patch_grid_shape"]
            )
            teacher_grid_shape = tuple(
                teacher_output["patch_grid_shape"]
            )

            if student_grid_shape != teacher_grid_shape:
                raise ValueError(
                    "Matching student and teacher views must have identical "
                    "patch_grid_shape values."
                )

            student_features = self._patches_to_grid(
                student_output["patch_features"],
                student_grid_shape,
            ).float()
            teacher_features = self._patches_to_grid(
                teacher_output["patch_features"].detach(),
                teacher_grid_shape,
            ).float()

            depth, height, width = student_grid_shape
            kernel = (
                min(self.region_pool_size[0], depth),
                min(self.region_pool_size[1], height),
                min(self.region_pool_size[2], width),
            )

            student_regions = F.avg_pool3d(
                student_features,
                kernel_size=kernel,
                stride=kernel,
            )
            teacher_regions = F.avg_pool3d(
                teacher_features,
                kernel_size=kernel,
                stride=kernel,
            )

            mask_grid = (
                mask.reshape(mask.shape[0], 1, depth, height, width)
                .float()
            )
            masked_fraction = F.avg_pool3d(
                mask_grid,
                kernel_size=kernel,
                stride=kernel,
            )
            valid_regions = (
                masked_fraction <= self.region_max_masked_fraction
            ).squeeze(1)

            student_regions = (
                student_regions
                .permute(0, 2, 3, 4, 1)
                .contiguous()
            )
            teacher_regions = (
                teacher_regions
                .permute(0, 2, 3, 4, 1)
                .contiguous()
            )

            if valid_regions.any():
                student_valid = F.normalize(
                    student_regions[valid_regions],
                    dim=-1,
                )
                teacher_valid = F.normalize(
                    teacher_regions[valid_regions],
                    dim=-1,
                )

                loss_terms.append(
                    (
                        1.0
                        - F.cosine_similarity(
                            student_valid,
                            teacher_valid,
                            dim=-1,
                        )
                    ).mean()
                )

        if not loss_terms:
            return student_global[0]["patch_features"].sum() * 0.0

        return torch.stack(loss_terms).mean()

    def compute_koleo_loss(
            self,
            student_global: Sequence[ModelOutput],
    ) -> torch.Tensor:
        """
        Compute KoLeo regularization independently for each global view.

        Keeping views separate prevents different augmentations of the same
        subject from becoming nearest neighbors.
        """
        if len(student_global) == 0:
            raise ValueError(
                "student_global must contain at least one global view."
            )

        loss_terms: List[torch.Tensor] = []

        for output in student_global:
            features = output["cls_features"]

            if features.ndim != 2:
                raise ValueError(
                    "cls_features must have shape [B, D], but received "
                    f"{tuple(features.shape)}."
                )

            if features.shape[0] < 2:
                continue

            # Compute KoLeo in float32 for numerical stability under AMP.
            normalized = F.normalize(
                features.float(),
                p=2,
                dim=-1,
                eps=self.koleo_eps,
            )

            distances = torch.cdist(
                normalized,
                normalized,
                p=2,
            )

            diagonal_mask = torch.eye(
                distances.shape[0],
                device=distances.device,
                dtype=torch.bool,
            )

            # Must be out of place: CdistBackward needs its original output.
            distances = distances.masked_fill(
                diagonal_mask,
                float("inf"),
            )

            nearest_neighbor_distance = distances.min(dim=1).values

            view_loss = -torch.log(
                nearest_neighbor_distance.clamp_min(self.koleo_eps)
            ).mean()

            loss_terms.append(view_loss)

        if not loss_terms:
            # Differentiable zero on the correct device.
            return student_global[0]["cls_features"].sum() * 0.0

        return torch.stack(loss_terms).mean()

    @torch.no_grad()
    def _distributed_batch_center(
        self,
        projections: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a globally synchronized mean projection."""
        flattened = projections.reshape(
            -1,
            projections.shape[-1],
        ).float()
        projection_sum = flattened.sum(dim=0, keepdim=True)
        projection_count = torch.tensor(
            [flattened.shape[0]],
            device=flattened.device,
            dtype=torch.float32,
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(projection_sum)
            dist.all_reduce(projection_count)

        return projection_sum / projection_count.clamp_min(1.0)

    @torch.no_grad()
    def update_objective_state(
        self,
        teacher_global: Sequence[ModelOutput],
    ) -> None:
        """Update the running DINO and iBOT teacher centers."""
        if not teacher_global:
            return

        self._ensure_centers(
            teacher_global[0]["cls_projection"],
            teacher_global[0]["patch_projection"],
        )

        cls_batch_center = self._distributed_batch_center(
            torch.cat(
                [
                    output["cls_projection"]
                    for output in teacher_global
                ],
                dim=0,
            )
        )
        patch_batch_center = self._distributed_batch_center(
            torch.cat(
                [
                    output["patch_projection"]
                    for output in teacher_global
                ],
                dim=0,
            )
        )

        self.dino_center.mul_(self.center_momentum).add_(
            cls_batch_center,
            alpha=1.0 - self.center_momentum,
        )
        self.ibot_center.mul_(self.center_momentum).add_(
            patch_batch_center,
            alpha=1.0 - self.center_momentum,
        )

    def _forward_model(
        self,
        model: nn.Module,
        x: torch.Tensor,
        info: Optional[Mapping[str, Any]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        info = dict(info or {})
        output = model(
            x,
            spacing=info.get("spacing"),
            modality=info.get("modality"),
            channel_mask=info.get("channel_mask"),
            mask=mask,
        )

        if not isinstance(output, dict):
            raise TypeError(
                "The model must return a dictionary, but returned "
                f"{type(output).__name__}."
            )

        missing = self.REQUIRED_OUTPUT_KEYS.difference(output.keys())
        if missing:
            raise KeyError(
                "The model output is missing required keys: "
                f"{sorted(missing)}."
            )

        self._validate_model_output(output)
        return output

    @staticmethod
    def _validate_model_output(output: ModelOutput) -> None:
        cls_features = output["cls_features"]
        patch_features = output["patch_features"]
        cls_projection = output["cls_projection"]
        patch_projection = output["patch_projection"]
        patch_grid_shape = output["patch_grid_shape"]

        for name, tensor in {
            "cls_features": cls_features,
            "patch_features": patch_features,
            "cls_projection": cls_projection,
            "patch_projection": patch_projection,
        }.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")

        if cls_features.ndim != 2:
            raise ValueError("cls_features must have shape [B, E].")
        if patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [B, N, E].")
        if cls_projection.ndim != 2:
            raise ValueError("cls_projection must have shape [B, K_cls].")
        if patch_projection.ndim != 3:
            raise ValueError(
                "patch_projection must have shape [B, N, K_patch]."
            )

        batch_size = cls_features.shape[0]
        for name, tensor in {
            "patch_features": patch_features,
            "cls_projection": cls_projection,
            "patch_projection": patch_projection,
        }.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} and cls_features must have equal batch sizes."
                )

        if patch_features.shape[1] != patch_projection.shape[1]:
            raise ValueError(
                "patch_features and patch_projection must contain the same "
                "number of tokens."
            )

        if not isinstance(patch_grid_shape, (tuple, list)):
            raise TypeError(
                "patch_grid_shape must be a tuple/list of three integers."
            )
        if len(patch_grid_shape) != 3:
            raise ValueError(
                "patch_grid_shape must contain exactly three dimensions."
            )
        if not all(isinstance(v, int) and v > 0 for v in patch_grid_shape):
            raise ValueError(
                "patch_grid_shape values must be positive integers."
            )

        expected_tokens = math.prod(patch_grid_shape)
        actual_tokens = patch_features.shape[1]
        if expected_tokens != actual_tokens:
            raise ValueError(
                "patch_grid_shape does not match token count: "
                f"{tuple(patch_grid_shape)} -> {expected_tokens}, "
                f"but patch_features has {actual_tokens}."
            )

    @staticmethod
    def _validate_batch_views(
        global_crops: Sequence[torch.Tensor],
        local_crops: Sequence[torch.Tensor],
        global_masks: Sequence[torch.Tensor],
        global_info: Sequence[ViewInfo],
        local_info: Sequence[ViewInfo],
    ) -> None:
        if len(global_crops) < 2:
            raise ValueError(
                "BrainDINO requires at least two global crops."
            )
        if len(global_masks) != len(global_crops):
            raise ValueError(
                "global_masks and global_crops must have equal lengths."
            )
        if len(global_info) != len(global_crops):
            raise ValueError(
                "global_info and global_crops must have equal lengths."
            )
        if len(local_info) != len(local_crops):
            raise ValueError(
                "local_info and local_crops must have equal lengths."
            )

        batch_size = global_crops[0].shape[0]

        for index, crop in enumerate(global_crops):
            if crop.ndim != 5:
                raise ValueError(
                    f"global_crops[{index}] must have shape [B, C, D, H, W]."
                )
            if crop.shape[0] != batch_size:
                raise ValueError(
                    "All global crops must have the same batch size."
                )

        for index, crop in enumerate(local_crops):
            if crop.ndim != 5:
                raise ValueError(
                    f"local_crops[{index}] must have shape [B, C, D, H, W]."
                )
            if crop.shape[0] != batch_size:
                raise ValueError(
                    "All local crops must use the global batch size."
                )

        for index, mask in enumerate(global_masks):
            if mask.dtype != torch.bool:
                raise TypeError(
                    f"global_masks[{index}] must be boolean."
                )
            if mask.ndim != 2:
                raise ValueError(
                    f"global_masks[{index}] must have shape [B, N]."
                )
            if mask.shape[0] != batch_size:
                raise ValueError(
                    "Every global mask must use the global batch size."
                )

    def _forward_all_views(
        self,
        batch: Mapping[str, Any],
    ) -> Tuple[
        List[ModelOutput],
        List[ModelOutput],
        List[ModelOutput],
    ]:
        global_crops: Sequence[torch.Tensor] = batch["global_crops"]
        teacher_global_crops: Sequence[torch.Tensor] = batch["teacher_global_crops"]
        local_crops: Sequence[torch.Tensor] = batch.get("local_crops", [])
        global_masks: Sequence[torch.Tensor] = batch["global_masks"]

        global_info: Sequence[ViewInfo] = batch.get(
            "global_info",
            [{} for _ in global_crops],
        )
        teacher_global_info: Sequence[ViewInfo] = batch.get(
            "teacher_global_info",
            [{} for _ in teacher_global_crops],
        )
        local_info: Sequence[ViewInfo] = batch.get(
            "local_info",
            [{} for _ in local_crops],
        )

        self._validate_batch_views(
            global_crops=global_crops,
            local_crops=local_crops,
            global_masks=global_masks,
            global_info=global_info,
            local_info=local_info,
        )

        student_global = [
            self._forward_model(
                self.model,
                crop,
                info=info,
                mask=mask,
            )
            for crop, info, mask in zip(
                global_crops,
                global_info,
                global_masks,
            )
        ]

        student_local = [
            self._forward_model(
                self.model,
                crop,
                info=info,
                mask=None,
            )
            for crop, info in zip(local_crops, local_info)
        ]

        with torch.no_grad():
            teacher_global = [
                self._forward_model(
                    self.teacher,
                    crop,
                    info=info,
                    mask=None,
                )
                for crop, info in zip(teacher_global_crops, teacher_global_info)
            ]

        return student_global, student_local, teacher_global

    def current_teacher_temperature(self) -> float:
        total_steps = self._total_optimizer_steps()
        warmup_steps = max(int(round(total_steps * self.warmup_ratio)), 1)

        progress = min(float(self.global_step) / float(warmup_steps), 1.0)
        return self.teacher_warmup_temperature + progress * (self.teacher_temperature - self.teacher_warmup_temperature)

    def current_teacher_momentum(self) -> float:
        total_steps = self._total_optimizer_steps()

        progress = min(float(self.global_step) / float(total_steps), 1.0)
        cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))

        return self.teacher_momentum + cosine_progress * (self.teacher_momentum_final - self.teacher_momentum)

    @torch.no_grad()
    def update_teacher(self) -> None:
        ema_momentum = self.current_teacher_momentum()
        student = self.unwrap_compiled_model()

        student_parameters = dict(student.named_parameters())
        teacher_parameters = dict(self.teacher.named_parameters())

        if student_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError(
                "Student and teacher parameter names do not match."
            )

        for name, teacher_parameter in teacher_parameters.items():
            student_parameter = student_parameters[name]
            teacher_parameter.data.mul_(ema_momentum).add_(
                student_parameter.data,
                alpha=1.0 - ema_momentum,
            )

        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(self.teacher.named_buffers())

        for name, teacher_buffer in teacher_buffers.items():
            student_buffer = student_buffers.get(name)
            if student_buffer is None:
                continue
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.data.mul_(ema_momentum).add_(
                    student_buffer.data,
                    alpha=1.0 - ema_momentum,
                )
            else:
                teacher_buffer.data.copy_(student_buffer.data)

    def on_before_zero_grad(self, optimizer) -> None:
        self.update_teacher()

    def compute_all_losses(
        self,
        batch: Mapping[str, Any],
        student_global: Sequence[ModelOutput],
        student_local: Sequence[ModelOutput],
        teacher_global: Sequence[ModelOutput],
        update_state: bool,
    ) -> Dict[str, torch.Tensor]:
        global_masks: Sequence[torch.Tensor] = batch["global_masks"]
        teacher_temperature = self.current_teacher_temperature()

        dino_loss = self.compute_dino_loss(
            student_global=student_global,
            student_local=student_local,
            teacher_global=teacher_global,
            teacher_temperature=teacher_temperature,
        )
        ibot_loss = self.compute_ibot_loss(
            student_global=student_global,
            teacher_global=teacher_global,
            global_masks=global_masks,
            teacher_temperature=teacher_temperature,
        )
        region_loss = self.compute_region_loss(
            student_global=student_global,
            teacher_global=teacher_global,
            global_masks=global_masks,
        )
        koleo_loss = self.compute_koleo_loss(
            student_global=student_global,
        )

        scalar_losses = {
            "dino": dino_loss,
            "ibot": ibot_loss,
            "region": region_loss,
            "koleo": koleo_loss,
        }
        for name, loss in scalar_losses.items():
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"{name} loss must be a torch.Tensor.")
            if loss.numel() != 1:
                raise ValueError(
                    f"{name} loss must be scalar, got {tuple(loss.shape)}."
                )

        total_loss = (
            self.dino_loss_weight * dino_loss
            + self.ibot_loss_weight * ibot_loss
            + self.region_loss_weight * region_loss
            + self.koleo_loss_weight * koleo_loss
        )

        if update_state:
            with torch.no_grad():
                self.update_objective_state(teacher_global=teacher_global)

        return {
            "total": total_loss,
            "dino": dino_loss,
            "ibot": ibot_loss,
            "region": region_loss,
            "koleo": koleo_loss,
            "teacher_temperature": total_loss.new_tensor(
                teacher_temperature
            ),
            "teacher_momentum": total_loss.new_tensor(
                self.current_teacher_momentum()
            ),
        }

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ):
        student_global, student_local, teacher_global = (
            self._forward_all_views(batch)
        )

        losses = self.compute_all_losses(
            batch=batch,
            student_global=student_global,
            student_local=student_local,
            teacher_global=teacher_global,
            update_state=True,
        )
        loss = losses["total"]

        if not torch.isfinite(loss):
            logging.error(
                "Non-finite BrainDINO loss at batch %s: %s",
                batch_idx,
                {
                    name: float(value.detach())
                    for name, value in losses.items()
                },
            )
            return None

        batch_size = batch["global_crops"][0].shape[0]

        self.log_dict(
            {
                "train/loss/total": losses["total"],
                "train/loss/dino": losses["dino"],
                "train/loss/ibot": losses["ibot"],
                "train/loss/region": losses["region"],
                "train/loss/koleo": losses["koleo"],
                "train/schedule/teacher_temperature": losses[
                    "teacher_temperature"
                ],
                "train/schedule/teacher_momentum": losses[
                    "teacher_momentum"
                ],
            },
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.global_step % self.log_every_n_steps == 0:
            self._log_training_metrics(
                batch=batch,
                loss=loss,
                student_global=student_global,
                batch_size=batch_size,
            )

            self.log_dict(
                self._compute_ssl_progress_metrics(
                    batch=batch,
                    student_global=student_global,
                    teacher_global=teacher_global,
                ),
                sync_dist=True,
                batch_size=batch_size,
            )

        self._log_wandb_visuals(
            batch=batch,
            student_global=student_global,
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ):
        student_global, student_local, teacher_global = (
            self._forward_all_views(batch)
        )

        losses = self.compute_all_losses(
            batch=batch,
            student_global=student_global,
            student_local=student_local,
            teacher_global=teacher_global,
            update_state=False,
        )

        batch_size = batch["global_crops"][0].shape[0]
        self.log_dict(
            {
                "val/loss/total": losses["total"],
                "val/loss/dino": losses["dino"],
                "val/loss/ibot": losses["ibot"],
                "val/loss/region": losses["region"],
                "val/loss/koleo": losses["koleo"],
            },
            sync_dist=True,
            batch_size=batch_size,
        )
        return losses["total"]

    @torch.no_grad()
    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        output = self._forward_model(
            self.model,
            batch["image"],
            info=batch.get("info", {}),
            mask=None,
        )
        return {
            "cls_features": output["cls_features"],
            "patch_features": output["patch_features"],
            "patch_grid_shape": output["patch_grid_shape"],
        }

    @torch.no_grad()
    def _log_training_metrics(
        self,
        batch: Mapping[str, Any],
        loss: torch.Tensor,
        student_global: Sequence[ModelOutput],
        batch_size: int,
    ) -> None:
        cls_features = torch.cat(
            [output["cls_features"] for output in student_global],
            dim=0,
        )
        patch_features = [
            output["patch_features"] for output in student_global
        ]

        metric_groups = {
            "features": feat_metrics.compute_train(cls_features),
            "performance": perf_metrics.compute(
                batch.get("transforms_applied"),
                batch_size,
            ),
            "stability": stability_metrics.compute_nan_inf_metrics(
                loss=loss,
                pred=cls_features,
                activations=patch_features,
            ),
        }

        self.log_dict(
            self._format_metrics(
                stage="train",
                metric_groups=metric_groups,
            ),
            sync_dist=True,
            batch_size=batch_size,
        )

    def _format_metrics(
        self,
        stage: str,
        metric_groups: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        metric_separator = "_" if self.mlflow_logging else "/"
        metrics: Dict[str, torch.Tensor] = {}
        for module_name, metric_dict in metric_groups.items():
            for key, value in metric_dict.items():
                metrics[
                    f"{stage}{metric_separator}{module_name}/{key}"
                ] = value
        return metrics

    def on_after_backward(self) -> None:
        grad_clip_val = getattr(
            self.trainer,
            "gradient_clip_val",
            None,
        )
        metric_groups = {
            "stability": stability_metrics.compute_on_backward(
                self.model,
                grad_clip_val,
            ),
            "performance": perf_metrics.compute_on_backward(self.trainer),
        }

        datamodule = getattr(self.trainer, "datamodule", None)
        batch_size = getattr(datamodule, "batch_size", None)

        self.log_dict(
            self._format_metrics(
                stage="train",
                metric_groups=metric_groups,
            ),
            sync_dist=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        decay_parameters = []
        no_decay_parameters = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            # Do not decay biases, normalization scales, or learned tokens.
            use_no_decay = (
                    parameter.ndim <= 1
                    or name.endswith(".bias")
                    or "cls_token" in name
                    or "mask_token" in name
                    or "position" in name.lower()
                    or "modality_token" in name
            )

            if use_no_decay:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)

        parameter_groups = [
            {
                "params": decay_parameters,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                parameter_groups,
                lr=self.learning_rate,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer}"
            )

        total_steps = self._total_optimizer_steps()

        scheduler = simple_warmup_cosine_decay_schedule(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_ratio=self.warmup_ratio,
            cosine_period_ratio=self.cosine_period_ratio,
            minimum_lr=self.minimum_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _define_wandb_visual_metrics(self) -> None:
        if self._wandb_visual_metrics_defined:
            return

        if not self.trainer.is_global_zero:
            return

        wandb_logger = self.get_logger_by_class_name("WandbLogger")
        if wandb_logger is None:
            return

        run = wandb_logger.experiment

        run.define_metric("visuals/global_step", hidden=True)
        run.define_metric("visuals/*", step_metric="visuals/global_step")

        self._wandb_visual_metrics_defined = True

    @staticmethod
    def _normalize_slice(image: torch.Tensor) -> torch.Tensor:
        image = image.float()

        finite = torch.isfinite(image)
        if not finite.any():
            return torch.zeros_like(image)

        values = image[finite]
        lower = torch.quantile(values, 0.01)
        upper = torch.quantile(values, 0.99)

        if upper <= lower:
            return torch.zeros_like(image)

        return (image.clamp(lower, upper) - lower) / (upper - lower)

    @classmethod
    def _volume_montage(
            cls,
            volume: torch.Tensor,
            mask_volume: Optional[torch.Tensor] = None,
            panel_size: tuple[int, int] = (192, 192),
    ) -> torch.Tensor:
        """
        Convert [D, H, W] into an RGB montage containing central
        axial, coronal, and sagittal slices.
        """
        if volume.ndim != 3:
            raise ValueError(
                f"Expected [D, H, W], got {tuple(volume.shape)}."
            )

        depth, height, width = volume.shape

        slices = [
            volume[depth // 2, :, :],
            volume[:, height // 2, :],
            volume[:, :, width // 2],
        ]

        if mask_volume is None:
            mask_slices = [None, None, None]
        else:
            mask_slices = [
                mask_volume[depth // 2, :, :],
                mask_volume[:, height // 2, :],
                mask_volume[:, :, width // 2],
            ]

        panels = []

        for image_slice, mask_slice in zip(slices, mask_slices, ):
            normalized = cls._normalize_slice(image_slice)

            normalized = F.interpolate(
                normalized[None, None],
                size=panel_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            rgb = normalized.unsqueeze(0).repeat(3, 1, 1)

            if mask_slice is not None:
                overlay = F.interpolate(
                    mask_slice.float()[None, None],
                    size=panel_size,
                    mode="nearest",
                )[0, 0].bool()

                rgb[0, overlay] = 1.0
                rgb[1, overlay] *= 0.25
                rgb[2, overlay] *= 0.25

            panels.append(rgb.cpu())

        return make_grid(panels, nrow=3, padding=4, normalize=False, pad_value=1.0)

    @staticmethod
    def _mask_to_volume(
            mask: torch.Tensor,
            patch_grid_shape: Sequence[int],
            spatial_shape: Sequence[int],
    ) -> torch.Tensor:
        """
        Convert a flattened token mask [N] to a voxel-resolution mask
        [D, H, W] for visualization.
        """
        patch_grid_shape = tuple(int(value) for value in patch_grid_shape)

        expected_tokens = math.prod(patch_grid_shape)
        if mask.numel() != expected_tokens:
            raise ValueError(
                f"Mask size does not match patch_grid_shape: {mask.numel()} versus {patch_grid_shape}."
            )

        token_mask = mask.reshape(1, 1, *patch_grid_shape).float()

        voxel_mask = F.interpolate(
            token_mask,
            size=tuple(int(value) for value in spatial_shape),
            mode="nearest",
        )

        return voxel_mask[0, 0].bool()

    @staticmethod
    def _binary_mask_montage(
            mask_volume: torch.Tensor,
            panel_size: tuple[int, int] = (192, 192),
    ) -> torch.Tensor:
        """
        Create an RGB montage of the central axial, coronal, and
        sagittal slices of a binary voxel mask.
        """
        if mask_volume.ndim != 3:
            raise ValueError(
                f"Expected [D, H, W], got {tuple(mask_volume.shape)}."
            )

        depth, height, width = mask_volume.shape

        mask_slices = [
            mask_volume[depth // 2, :, :],
            mask_volume[:, height // 2, :],
            mask_volume[:, :, width // 2],
        ]

        panels = []

        for mask_slice in mask_slices:
            resized = F.interpolate(
                mask_slice.float()[None, None],
                size=panel_size,
                mode="nearest",
            )[0, 0]

            rgb = resized.unsqueeze(0).repeat(3, 1, 1)
            panels.append(rgb.cpu())

        return make_grid(
            panels,
            nrow=3,
            padding=4,
            normalize=False,
            pad_value=1.0,
        )

    @classmethod
    def _global_mask_comparison(
            cls,
            volume: torch.Tensor,
            mask_volume: torch.Tensor,
            panel_size: tuple[int, int] = (192, 192),
    ) -> torch.Tensor:
        """
        Create a vertical comparison:

            Row 1: original axial, coronal, sagittal slices
            Row 2: the same slices with masked tokens highlighted
            Row 3: the binary token mask
        """
        original = cls._volume_montage(
            volume=volume,
            panel_size=panel_size,
        )

        overlay = cls._volume_montage(
            volume=volume,
            mask_volume=mask_volume,
            panel_size=panel_size,
        )

        binary_mask = cls._binary_mask_montage(
            mask_volume=mask_volume,
            panel_size=panel_size,
        )

        return make_grid(
            [original, overlay, binary_mask],
            nrow=1,
            padding=8,
            normalize=False,
            pad_value=1.0,
        )

    @torch.no_grad()
    def _log_wandb_visuals(
            self,
            batch: Mapping[str, Any],
            student_global: Sequence[ModelOutput],
    ) -> None:
        if not self.trainer.is_global_zero:
            return

        if self.visual_log_every_n_steps == 0:
            return

        step = int(self.global_step)

        # Avoid duplicate logs across gradient-accumulation microbatches.
        if step == self._last_visual_log_step:
            return

        if step % self.visual_log_every_n_steps != 0:
            return

        wandb_logger = self.get_logger_by_class_name("WandbLogger")
        if wandb_logger is None:
            return

        self._define_wandb_visual_metrics()

        global_crops = batch["global_crops"]
        teacher_global_crops = batch["teacher_global_crops"]
        local_crops = batch.get("local_crops", [])
        global_masks = batch["global_masks"]

        sample_index = self.visual_log_sample_index
        batch_size = global_crops[0].shape[0]

        if sample_index >= batch_size:
            return

        teacher_images = []
        global_comparison_images = []
        local_images = []

        def add_crop(
                destination: list,
                crop: torch.Tensor,
                view_name: str,
        ) -> None:
            sample = crop[sample_index].detach()
            number_channels = min(sample.shape[0], self.visual_log_max_channels)

            for channel_index in range(number_channels):
                montage = self._volume_montage(volume=sample[channel_index])

                destination.append(
                    wandb.Image(
                        montage,
                        caption=(
                            f"view={view_name} | "
                            f"channel={channel_index} | "
                            f"sample={sample_index} | "
                            f"optimizer_step={step}"
                        ),
                    )
                )

        # Teacher global crops are useful for checking teacher preprocessing.
        for view_index, crop in enumerate(teacher_global_crops):
            add_crop(
                destination=teacher_images,
                crop=crop,
                view_name=f"teacher_global_{view_index}",
            )

        # Combine original, mask overlay, and binary mask in one image.
        for view_index, crop in enumerate(global_crops):
            sample = crop[sample_index].detach()

            mask_volume = self._mask_to_volume(
                mask=global_masks[view_index][sample_index],
                patch_grid_shape=student_global[
                    view_index
                ]["patch_grid_shape"],
                spatial_shape=crop.shape[-3:],
            )

            number_channels = min(sample.shape[0], self.visual_log_max_channels)

            for channel_index in range(number_channels):
                comparison = self._global_mask_comparison(
                    volume=sample[channel_index],
                    mask_volume=mask_volume,
                )

                global_comparison_images.append(
                    wandb.Image(
                        comparison,
                        caption=(
                            f"global_view={view_index} | "
                            f"channel={channel_index} | "
                            f"sample={sample_index} | "
                            f"optimizer_step={step} | "
                            "rows=original, mask_overlay, binary_mask"
                        ),
                    )
                )

        # Local crops do not have iBOT masks.
        for view_index, crop in enumerate(local_crops):
            add_crop(
                destination=local_images,
                crop=crop,
                view_name=f"student_local_{view_index}",
            )

        visual_data = {"visuals/global_step": step}

        if global_comparison_images:
            visual_data["visuals/global_mask_comparison"] = global_comparison_images

        if teacher_images:
            visual_data["visuals/teacher_global"] = teacher_images

        if local_images:
            visual_data["visuals/student_local"] = local_images

        wandb_logger.experiment.log(visual_data, commit=True,)

        self._last_visual_log_step = step

    @torch.no_grad()
    def _compute_ssl_progress_metrics(
            self,
            batch: Mapping[str, Any],
            student_global: Sequence[ModelOutput],
            teacher_global: Sequence[ModelOutput],
    ) -> Dict[str, torch.Tensor]:
        """
            cls_feature_std approaching zero indicates collapse.
            mean_pairwise_cosine approaching 1.0 indicates all samples are becoming nearly identical.
            teacher_student_cosine should generally increase, but instantly reaching almost 1.0 alongside low feature standard is suspicious.
            Center norms should remain finite and change smoothly.
            mask_ratio verifies that the intended masking configuration reaches the loss.
        """
        student_cls = torch.cat([output["cls_features"].float() for output in student_global], dim=0)
        teacher_cls = torch.cat([output["cls_features"].float() for output in teacher_global], dim=0)

        student_normalized = F.normalize(student_cls, dim=-1)
        teacher_normalized = F.normalize(teacher_cls, dim=-1)

        feature_std = student_cls.std(dim=0).mean()

        centered = student_cls - student_cls.mean(dim=0)
        feature_rms = centered.square().mean().sqrt()

        if student_normalized.shape[0] > 1:
            similarity = (student_normalized @ student_normalized.transpose(0, 1))
            off_diagonal = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
            mean_pairwise_cosine = similarity[off_diagonal].mean()
        else:
            mean_pairwise_cosine = student_cls.new_tensor(0.0)

        matched_student = torch.stack([output["cls_features"].float() for output in student_global], dim=0)
        matched_teacher = torch.stack([output["cls_features"].float() for output in teacher_global], dim=0)

        teacher_student_cosine = F.cosine_similarity(matched_student, matched_teacher, dim=-1).mean()

        mask_ratio = torch.cat([mask.float().reshape(-1) for mask in batch["global_masks"]]).mean()

        return {
            "ssl_progress/cls_feature_std": feature_std,
            "ssl_progress/cls_feature_rms": feature_rms,
            "ssl_progress/mean_pairwise_cosine": mean_pairwise_cosine,
            "ssl_progress/teacher_student_cosine": teacher_student_cosine,
            "ssl_progress/mask_ratio": mask_ratio,
            "ssl_progress/dino_center_norm": self.dino_center.float().norm(),
            "ssl_progress/ibot_center_norm": self.ibot_center.float().norm(),
        }