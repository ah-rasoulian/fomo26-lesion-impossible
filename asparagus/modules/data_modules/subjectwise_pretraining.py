from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import Compose

from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.SubjectWisePretrainDataset import (
    SubjectWisePretrainDataset,
)
from asparagus.modules.datasets.TrainDataset import SingleSubjectPredictDataset
from asparagus.modules.transforms.presets import (
    braindino_CPU_train_transforms,
    braindino_CPU_val_transforms,
)


class ReplacementDistributedSampler(DistributedSampler):
    """Deterministic replacement sampling with equal work on every rank."""

    def __init__(self, dataset, global_num_samples: int, seed: int = 0) -> None:
        if global_num_samples < 1:
            raise ValueError("global_num_samples must be positive.")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("This sampler requires initialized DDP.")
        super().__init__(
            dataset=dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        self.num_samples = math.ceil(global_num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randint(
            len(self.dataset),
            (self.total_size,),
            generator=generator,
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("Distributed replacement sampling was uneven.")
        return iter(indices)


def _vector(
    value: Any,
    length: int,
    name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).flatten()
    if tensor.numel() != length:
        raise ValueError(
            f"{name} must contain {length} values, got {tuple(tensor.shape)}."
        )
    return tensor


def _aggregate_transform_counts(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """Combine per-subject augmentation records for legacy rate metrics."""
    counts: Dict[str, float] = {}
    for sample in samples:
        applied = sample.get("transforms_applied") or {}
        if not isinstance(applied, Mapping):
            raise TypeError(
                "Each sample's transforms_applied entry must be a mapping, "
                f"got {type(applied).__name__}."
            )
        for name, value in applied.items():
            if torch.is_tensor(value):
                value = value.detach().sum().item()
            elif isinstance(value, (list, tuple)):
                value = sum(float(item) for item in value)
            key = str(name)
            counts[key] = counts.get(key, 0.0) + float(value)
    return counts


def subjectwise_full_volume_collate(
    samples: Sequence[Mapping[str, Any]],
    unknown_modality_id: int = 0,
) -> Dict[str, Any]:
    """
    End-pad variable full volumes and channel counts for a raw batch.

    The true per-subject shape is stored in ``valid_spatial_shapes``. The GPU
    view transform and physical convolution use it to ignore batching padding.
    No crop, resize, pooling, or spacing change occurs here.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    images = []
    infos = []
    for index, sample in enumerate(samples):
        image = sample.get("image")
        info = sample.get("info")
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            shape = getattr(image, "shape", None)
            raise ValueError(
                f"samples[{index}]['image'] must be [C, H, W, D], got {shape}."
            )
        if not isinstance(info, Mapping):
            raise TypeError(f"samples[{index}]['info'] must be a mapping.")
        if image.shape[0] < 1 or any(size < 1 for size in image.shape[-3:]):
            raise ValueError(f"samples[{index}] contains an empty dimension.")
        images.append(image.contiguous())
        infos.append(info)

    max_channels = max(image.shape[0] for image in images)
    max_shape = tuple(
        max(image.shape[axis] for image in images) for axis in range(1, 4)
    )
    batch = images[0].new_zeros((len(images), max_channels, *max_shape))
    channel_mask = torch.zeros(len(images), max_channels, dtype=torch.bool)
    modality = torch.full(
        (len(images), max_channels),
        int(unknown_modality_id),
        dtype=torch.long,
    )
    valid_shapes = torch.empty(len(images), 3, dtype=torch.long)
    spacing = torch.empty(len(images), 3, dtype=torch.float32)

    for batch_index, (image, info) in enumerate(zip(images, infos)):
        channels, height, width, depth = image.shape
        batch[
            batch_index,
            :channels,
            :height,
            :width,
            :depth,
        ] = image
        valid_shapes[batch_index] = torch.tensor((height, width, depth))

        if "spacing" not in info:
            raise KeyError(f"samples[{batch_index}]['info'] lacks spacing.")
        sample_spacing = _vector(
            info["spacing"], 3, "spacing", torch.float32
        )
        if not torch.isfinite(sample_spacing).all() or (
            sample_spacing <= 0
        ).any():
            raise ValueError("All spacing values must be finite and positive.")
        spacing[batch_index] = sample_spacing

        sample_modality = _vector(
            info.get("modality", [unknown_modality_id] * channels),
            channels,
            "modality",
            torch.long,
        )
        sample_mask = _vector(
            info.get("channel_mask", [True] * channels),
            channels,
            "channel_mask",
            torch.bool,
        )
        if not sample_mask.any():
            raise ValueError("Every subject must retain at least one channel.")
        modality[batch_index, :channels] = sample_modality
        channel_mask[batch_index, :channels] = sample_mask

    return {
        "image": batch,
        "info": {
            "spacing": spacing,
            "modality": modality,
            "channel_mask": channel_mask,
            "valid_spatial_shapes": valid_shapes,
        },
        # performance.compute expects batch-level augmentation counts rather
        # than a list of per-subject dictionaries.
        "transforms_applied": _aggregate_transform_counts(samples),
        "session_id": [sample.get("session_id") for sample in samples],
        "session_path": [sample.get("session_path") for sample in samples],
    }


class SubjectWisePretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        predict_samples: Optional[list] = None,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
        num_samples: Optional[int] = None,
        max_channels: int = 4,
        unknown_modality_id: int = 0,
        train_metadata_cache_path: Optional[str] = None,
        val_metadata_cache_path: Optional[str] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        sampler_seed: int = 0,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive when provided.")
        if max_channels < 1:
            raise ValueError("max_channels must be positive.")
        if unknown_modality_id < 0:
            raise ValueError("unknown_modality_id must be non-negative.")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        if (
            train_metadata_cache_path is not None
            and train_metadata_cache_path == val_metadata_cache_path
        ):
            raise ValueError("Training and validation caches must differ.")

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_split = train_split
        self.val_split = val_split
        self.predict_samples = predict_samples or []
        self.train_transforms = (
            train_transforms
            if train_transforms is not None
            else braindino_CPU_train_transforms()
        )
        self.val_transforms = (
            val_transforms
            if val_transforms is not None
            else braindino_CPU_val_transforms()
        )
        self.predict_transforms = predict_transforms
        self.num_samples = num_samples
        self.max_channels = int(max_channels)
        self.unknown_modality_id = int(unknown_modality_id)
        self.train_metadata_cache_path = train_metadata_cache_path
        self.val_metadata_cache_path = val_metadata_cache_path
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory)
        self.sampler_seed = int(sampler_seed)
        logging.info("Using %d data-loader workers", self.num_workers)

    def prepare_data(self) -> None:
        if self.train_metadata_cache_path is not None:
            SubjectWisePretrainDataset(
                self.train_split,
                transforms=None,
                max_channels=self.max_channels,
                metadata_cache_path=self.train_metadata_cache_path,
            )
        if self.val_metadata_cache_path is not None:
            SubjectWisePretrainDataset(
                self.val_split,
                transforms=None,
                max_channels=self.max_channels,
                is_validation=True,
                metadata_cache_path=self.val_metadata_cache_path,
            )

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ) -> None:
        if stage in (None, "fit", "validate"):
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage is not supported.")
        elif stage == "predict":
            self.setup_predict()

    def setup_fit(self) -> None:
        if not hasattr(self, "train_dataset"):
            self.train_dataset = SubjectWisePretrainDataset(
                self.train_split,
                transforms=self.train_transforms,
                max_channels=self.max_channels,
                metadata_cache_path=self.train_metadata_cache_path,
                show_progress=False,
            )
        if not hasattr(self, "val_dataset"):
            self.val_dataset = SubjectWisePretrainDataset(
                self.val_split,
                transforms=self.val_transforms,
                max_channels=self.max_channels,
                is_validation=True,
                metadata_cache_path=self.val_metadata_cache_path,
                show_progress=False,
            )

    def setup_predict(self) -> None:
        if not hasattr(self, "predict_dataset"):
            self.predict_dataset = SingleSubjectPredictDataset(
                self.predict_samples,
                transforms=self.predict_transforms,
            )

    def _worker_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": (
                self.persistent_workers and self.num_workers > 0
            ),
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        global_num_samples = self.num_samples or len(self.train_dataset)
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if math.ceil(global_num_samples / world_size) < self.batch_size:
            raise ValueError(
                "num_samples produces fewer than one full batch per rank."
            )
        if world_size > 1:
            sampler = ReplacementDistributedSampler(
                self.train_dataset,
                global_num_samples,
                seed=self.sampler_seed,
            )
        else:
            generator = torch.Generator().manual_seed(self.sampler_seed)
            sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=global_num_samples,
                generator=generator,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=partial(
                subjectwise_full_volume_collate,
                unknown_modality_id=self.unknown_modality_id,
            ),
            **self._worker_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        sampler = (
            DistributedSampler(
                self.val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False,
            )
            if dist.is_available() and dist.is_initialized()
            else None
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(
                subjectwise_full_volume_collate,
                unknown_modality_id=self.unknown_modality_id,
            ),
            **self._worker_kwargs(),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_return,
            **self._worker_kwargs(),
        )
