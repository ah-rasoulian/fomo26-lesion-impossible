import lightning as pl
import logging
import torch.distributed as dist
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import (
    ClsRegDataset,
    ClsRegTestDataset,
    SegDataset,
    SegTestDataset,
    SingleSubjectPredictDataset,
)
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, RandomSampler
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
        test_samples: list = [],
        predict_samples: Optional[list] = [],
        predict_transforms: Optional[Compose] = None,
        train_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.test_samples = test_samples
        self.val_split = val_split
        self.predict_samples = predict_samples
        self.predict_transforms = predict_transforms

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
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

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            persistent_workers=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collate_return,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=collate_return,
        )


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
        log_split_details: bool = True,
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
        self.predict_samples = predict_samples or []
        self.predict_transforms = predict_transforms
        self.log_split_details = log_split_details

        logging.info("Using %d workers", self.num_workers)

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            self.setup_test()
        elif stage == "predict":
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
        if self.use_random_datasampler:
            sampler = RandomSampler(self.train_dataset, num_samples=999999, replacement=True)
            sampler = DistributedSamplerWrapper(sampler) if dist.is_initialized() else sampler

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
            shuffle=sampler is None,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            persistent_workers=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=1,
            batch_size=1,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collate_return,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=collate_return,
        )


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
