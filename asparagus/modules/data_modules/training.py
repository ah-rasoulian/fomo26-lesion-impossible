import lightning as pl
import logging
import torch.distributed as dist
from asparagus.modules.datasets.TrainDataset import (
    ClsRegDataset,
    ClsRegTestDataset,
    SegDataset,
    SegTestDataset,
    SingleSubjectPredictDataset,
    full_volume_task_collate,
)
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torchvision.transforms import Compose
from typing import Literal, Optional
from collections import Counter
from pathlib import Path
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class SegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        test_samples: Optional[list] = None,
        predict_samples: Optional[list] = None,
        predict_transforms: Optional[Compose] = None,
        train_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.test_samples = list(test_samples or [])
        self.val_split = val_split
        self.predict_samples = list(predict_samples or [])
        self.predict_transforms = predict_transforms
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = (
            self.num_workers > 0
            if persistent_workers is None
            else bool(persistent_workers)
        )
        self.prefetch_factor = int(prefetch_factor)
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0.")

        logging.info(f"Using {self.num_workers} workers")

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        if stage in (None, "fit", "validate"):
            self.setup_fit()
        if stage in (None, "test"):
            self.setup_test()
        if stage in (None, "predict"):
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = SegDataset(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = SegDataset(
            self.val_split,
            transforms=self.val_transforms,
        )

    def setup_test(self):
        self.test_dataset = SegTestDataset(
            self.test_samples,
            transforms=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = SingleSubjectPredictDataset(
            self.predict_samples,
            transforms=self.predict_transforms,
        )

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, num_samples=999999, replacement=True)
        if dist.is_initialized():
            sampler = DistributedSamplerWrapper(sampler)

        return self._loader(
            self.train_dataset, batch_size=self.batch_size,
            drop_last=True, sampler=sampler,
        )

    def val_dataloader(self):
        return self._loader(
            self.val_dataset, batch_size=self.batch_size,
            drop_last=False, shuffle=False,
        )

    def test_dataloader(self):
        return self._loader(self.test_dataset, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        return self._loader(self.predict_dataset, batch_size=1, shuffle=False)

    def _loader(self, dataset, **kwargs):
        loader_kwargs = {
            "dataset": dataset,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": full_volume_task_collate,
            **kwargs,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_kwargs)


class ClsRegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        predict_transforms: Optional[Compose] = None,
        test_samples: Optional[list] = None,
        predict_samples: Optional[list] = None,
        use_random_datasampler: bool = True,
        use_weighted_sampler: bool = False,
        weighted_sampler_power: float = 1.0,
        train_num_samples: Optional[int] = None,
        sampler_seed: int = 0,
        log_split_details: bool = True,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_samples = test_samples or []
        self.use_random_datasampler = use_random_datasampler
        self.use_weighted_sampler = bool(use_weighted_sampler)
        self.weighted_sampler_power = float(weighted_sampler_power)
        self.train_num_samples = train_num_samples
        self.sampler_seed = int(sampler_seed)
        if self.use_random_datasampler and self.use_weighted_sampler:
            raise ValueError(
                "use_random_datasampler and use_weighted_sampler are mutually exclusive."
            )
        if not 0.0 < self.weighted_sampler_power <= 1.0:
            raise ValueError("weighted_sampler_power must be in (0, 1].")
        if self.train_num_samples is not None and self.train_num_samples < 1:
            raise ValueError("train_num_samples must be positive when provided.")
        self.predict_samples = predict_samples or []
        self.predict_transforms = predict_transforms
        self.log_split_details = log_split_details
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = (
            self.num_workers > 0
            if persistent_workers is None
            else bool(persistent_workers)
        )
        self.prefetch_factor = int(prefetch_factor)
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0.")

        logging.info("Using %d workers", self.num_workers)

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        if stage in (None, "fit", "validate"):
            self.setup_fit()
        if stage in (None, "test"):
            self.setup_test()
        if stage in (None, "predict"):
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = ClsRegDataset(
            self.train_split,
            transforms=self.train_transforms,
        )

        self.val_dataset = ClsRegDataset(
            self.val_split,
            transforms=self.val_transforms,
        )

        if self.log_split_details:
            self._log_fit_split()

    @staticmethod
    def _read_clsreg_label(file: str):
        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(data, (tuple, list)) or len(data) < 2:
            raise RuntimeError(
                f"Expected (image, label) in {file}, but received "
                f"{type(data).__name__}."
            )

        label = torch.as_tensor(data[1]).reshape(-1)

        if label.numel() == 1:
            return label.item()

        return tuple(label.tolist())

    @staticmethod
    def _subject_id(file: str) -> str:
        return Path(file).parent.parent.name

    @rank_zero_only
    def _log_fit_split(self) -> None:
        train_files = [str(file) for file in self.train_split]
        val_files = [str(file) for file in self.val_split]

        train_set = set(train_files)
        val_set = set(val_files)
        overlap = sorted(train_set.intersection(val_set))

        if overlap:
            raise RuntimeError(
                "Files occur in both training and validation splits: "
                f"{overlap}"
            )

        train_records = [
            (file, self._read_clsreg_label(file))
            for file in train_files
        ]
        val_records = [
            (file, self._read_clsreg_label(file))
            for file in val_files
        ]

        logging.info("=" * 80)
        logging.info("CLASSIFICATION/REGRESSION DATA SPLIT")
        logging.info(
            "Training cases: %d | Validation cases: %d",
            len(train_records),
            len(val_records),
        )

        logging.info("Training label counts: %s", dict(
            sorted(Counter(label for _, label in train_records).items())
        ))
        logging.info("Validation label counts: %s", dict(
            sorted(Counter(label for _, label in val_records).items())
        ))

        logging.info("TRAINING CASES")
        for index, (file, label) in enumerate(train_records):
            logging.info(
                "  train[%03d] label=%s file=%s",
                index,
                label,
                file,
            )

        logging.info("VALIDATION CASES")
        for index, (file, label) in enumerate(val_records):
            logging.info(
                "  val[%03d] label=%s file=%s",
                index,
                label,
                file,
            )

        logging.info("=" * 80)

        train_subjects = {
            self._subject_id(file)
            for file in train_files
        }
        val_subjects = {
            self._subject_id(file)
            for file in val_files
        }

        subject_overlap = sorted(
            train_subjects.intersection(val_subjects)
        )

        if subject_overlap:
            raise RuntimeError(
                "Subjects occur in both training and validation splits: "
                f"{subject_overlap}"
            )

        logging.info(
            "Training subjects: %d | Validation subjects: %d",
            len(train_subjects),
            len(val_subjects),
        )

    def get_class_counts(
            self,
            num_classes: int,
    ) -> torch.Tensor:
        """Return training-set sample counts for every class."""
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        counts = torch.zeros(num_classes, dtype=torch.long)

        for file in self.train_split:
            label = self._read_clsreg_label(str(file))

            if isinstance(label, tuple):
                raise ValueError(
                    "Class-weight calculation expects one scalar class label "
                    f"per sample, but {file} has label {label}."
                )

            # Reject non-integer regression targets accidentally supplied to a
            # classification experiment.
            numeric_label = float(label)
            class_index = int(numeric_label)

            if numeric_label != class_index:
                raise ValueError(
                    f"Expected an integer class label in {file}, got {label}."
                )

            if not 0 <= class_index < num_classes:
                raise ValueError(
                    f"Class label {class_index} in {file} is outside the "
                    f"valid range [0, {num_classes - 1}]."
                )

            counts[class_index] += 1

        missing_classes = torch.where(counts == 0)[0].tolist()
        if missing_classes:
            raise RuntimeError(
                "Cannot calculate finite class weights because the training "
                f"split contains no samples for classes {missing_classes}."
            )

        return counts

    def compute_class_weights(
            self,
            num_classes: int,
            power: float = 0.5,
    ) -> torch.Tensor:
        """Compute normalized inverse-frequency class weights.

        power=0.0:
            No weighting.

        power=0.5:
            Square-root inverse-frequency weighting. Recommended for very small
            datasets because it is less aggressive.

        power=1.0:
            Standard balanced inverse-frequency weighting:
            N / (number_of_classes * class_count).
        """
        if not 0.0 <= power <= 1.0:
            raise ValueError("power must be in [0, 1].")

        counts = self.get_class_counts(num_classes).float()
        total = counts.sum()

        balanced_weights = total / (num_classes * counts)
        weights = balanced_weights.pow(power)

        # Keep the average class weight at one. This makes the overall loss scale
        # comparable across weighted and unweighted experiments.
        weights = weights / weights.mean()

        logging.info(
            "Training class counts: %s",
            counts.long().tolist(),
        )
        logging.info(
            "Cross-entropy class weights (power=%.3f): %s",
            power,
            [round(value, 6) for value in weights.tolist()],
        )

        return weights

    def setup_test(self):
        self.test_dataset = ClsRegTestDataset(
            self.test_samples,
            transforms=self.test_transforms,
        )

    def setup_predict(self):
        self.predict_dataset = SingleSubjectPredictDataset(
            self.predict_samples,
            transforms=self.predict_transforms,
        )

    def train_dataloader(self):
        sampler = None

        number_of_samples = self.train_num_samples or len(self.train_dataset)
        generator = torch.Generator().manual_seed(self.sampler_seed)

        if self.use_weighted_sampler:
            labels = []
            for file in self.train_split:
                label = self._read_clsreg_label(str(file))
                if isinstance(label, tuple):
                    raise ValueError(
                        "Weighted sampling requires one scalar class label per sample."
                    )
                numeric_label = float(label)
                class_index = int(numeric_label)
                if numeric_label != class_index or class_index < 0:
                    raise ValueError(
                        f"Weighted sampling requires non-negative integer labels; "
                        f"{file} contains {label}."
                    )
                labels.append(class_index)

            counts = Counter(labels)
            sample_weights = torch.tensor(
                [counts[label] ** (-self.weighted_sampler_power) for label in labels],
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=number_of_samples,
                replacement=True,
                generator=generator,
            )
            logging.info(
                "Using weighted replacement sampling: counts=%s, power=%.3f, "
                "samples_per_epoch=%d",
                dict(sorted(counts.items())),
                self.weighted_sampler_power,
                number_of_samples,
            )
        elif self.use_random_datasampler:
            sampler = RandomSampler(
                self.train_dataset,
                num_samples=number_of_samples,
                replacement=True,
                generator=generator,
            )

            if dist.is_initialized():
                sampler = DistributedSamplerWrapper(sampler)

        loader_kwargs = {
            "dataset": self.train_dataset,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": True,
            "shuffle": sampler is None,
            "sampler": sampler,
            "collate_fn": full_volume_task_collate,
        }

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = {
            "dataset": self.val_dataset,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": full_volume_task_collate,
        }

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**loader_kwargs)

    def test_dataloader(self):
        loader_kwargs = {
            "dataset": self.test_dataset,
            "num_workers": self.num_workers,
            "batch_size": 1,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": full_volume_task_collate,
        }

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**loader_kwargs)

    def predict_dataloader(self):
        loader_kwargs = {
            "dataset": self.predict_dataset,
            "num_workers": self.num_workers,
            "batch_size": 1,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": full_volume_task_collate,
        }

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**loader_kwargs)


if __name__ == "__main__":
    from asparagus.functional.loading import load_json

    dataset_json = load_json(
        "/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task997_LauritSynSeg/dataset.json"
    )
    splits = load_json(
        "/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task997_LauritSynSeg/split_80_20.json"
    )[0]
    train_split = splits["train"]
    val_split = splits["val"]
    data_module = SegDataModule(
        train_split=train_split,
        val_split=val_split,
        batch_size=2,
        num_workers=6,
    )
    data_module.setup("fit")
    data_module_iterator = iter(data_module.train_dataloader())
    x = next(data_module_iterator)
    print(type(x))
    print(next(iter(data_module.train_dataset))["image"].shape)
