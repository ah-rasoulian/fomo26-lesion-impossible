from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import torchvision
from nibabel.orientations import aff2axcodes
from torch.utils.data import Dataset

from asparagus.functional.loading import MODALITY_TO_ID, get_modality_id
from asparagus.paths import get_data_path, get_source_labels_path
from gardening_tools.functional.nibabel_utils import reorient_nib_image
from gardening_tools.functional.paths.read import (
    load_pickle,
    read_file_to_nifti_or_np,
)
from gardening_tools.functional.type_conversions import nifti_or_np_to_np


def get_processed_data_info(file: str) -> dict:
    properties_file = file.replace(".pt", ".pkl")
    properties = load_pickle(properties_file)

    unknown_id = int(MODALITY_TO_ID.get("UNKNOWN", 0))
    modality_ids = torch.tensor(
        [MODALITY_TO_ID.get(modality, unknown_id) for modality in properties["modalities"]],
        dtype=torch.long,
    )

    spacing = torch.as_tensor(properties["new_spacing"], dtype=torch.float32)
    if spacing.shape != (3,) or not torch.isfinite(spacing).all() or (spacing <= 0).any():
        raise RuntimeError(f"Invalid [H, W, D] spacing in metadata for {file}: {spacing}.")

    return {
        "affine": torch.as_tensor(properties["nifti_metadata"]["affine"], dtype=torch.float32),
        "spacing": spacing,
        "direction": properties["new_direction"],
        "modality": modality_ids,
    }


def _ensure_image_shape(
    image: torch.Tensor,
    file: str,
) -> torch.Tensor:
    """
    Ensure that an image follows the [C, H, W, D] convention.
    """
    image = torch.as_tensor(image).float()

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.ndim != 4:
        raise RuntimeError(
            f"Expected image with shape [C, H, W, D], but {file} "
            f"produced shape {tuple(image.shape)}."
        )

    return image


def _ensure_label_shape(
    label: torch.Tensor,
    file: str,
) -> torch.Tensor:
    """
    Ensure that a segmentation label follows [C, H, W, D].
    """
    label = torch.as_tensor(label).float()

    if label.ndim == 3:
        label = label.unsqueeze(0)

    if label.ndim != 4:
        raise RuntimeError(
            f"Expected segmentation label with shape [C, H, W, D], "
            f"but {file} produced shape {tuple(label.shape)}."
        )

    return label


def _validate_modality_count(
    data_dict: dict,
    file: str,
) -> None:
    """
    Verify that one modality ID is provided for each image channel.
    """
    image = data_dict.get("image")
    info = data_dict.get("info", {})
    modality = info.get("modality")

    if image is None or modality is None:
        return

    modality = torch.as_tensor(modality,dtype=torch.long).reshape(-1)

    info["modality"] = modality

    if modality.numel() != image.shape[0]:
        raise RuntimeError(
            f"Image/modality mismatch for {file}: image has "
            f"{image.shape[0]} channels, but info['modality'] has "
            f"{modality.numel()} entries."
        )

    info["channel_mask"] = torch.ones(image.shape[0], dtype=torch.bool)
    info.setdefault(
        "valid_spatial_shapes",
        torch.tensor(image.shape[1:], dtype=torch.long),
    )


def _finalize_sample_metadata(data_dict: dict, file: str) -> dict:
    """Finalize metadata after transforms without treating padding as anatomy."""
    image = _ensure_image_shape(data_dict["image"], file)
    data_dict["image"] = image
    info = data_dict.setdefault("info", {})

    current_shape = torch.tensor(image.shape[1:], dtype=torch.long)
    previous_shape = info.get("valid_spatial_shapes")
    if previous_shape is None:
        valid_shape = current_shape
    else:
        previous_shape = torch.as_tensor(previous_shape, dtype=torch.long).reshape(3)
        valid_shape = torch.minimum(previous_shape, current_shape)
    if (valid_shape <= 0).any():
        raise RuntimeError(f"Invalid spatial shape for {file}: {valid_shape.tolist()}.")
    info["valid_spatial_shapes"] = valid_shape

    _validate_modality_count(data_dict, file)
    return data_dict


def _sanitize_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor

    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=4.0,
            neginf=-1.0,
        )

    return tensor


def _sanitize_data_dict(
    data_dict: dict,
) -> dict:
    """
    Sanitize image and task labels after all transforms have been applied.
    """
    tensor_keys = (
        "image",
        "label",
        "SEG_label",
        "CLSREG_label",
        "src_label",
    )

    for key in tensor_keys:
        value = data_dict.get(key)

        if isinstance(value, torch.Tensor):
            data_dict[key] = _sanitize_tensor(value)

    return data_dict


class BaseTaskDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms

        # Retained for compatibility with code that accesses this name.
        self.composed_transforms = transforms

    def __len__(self):
        return len(self.files)

    def _transform(
        self,
        data_dict: dict,
    ) -> dict:
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        data_dict = _sanitize_data_dict(data_dict)
        data_dict = _finalize_sample_metadata(
            data_dict, str(data_dict.get("file_path", "sample"))
        )
        if "label" in data_dict:
            data_dict["SEG_label"] = data_dict["label"]
        return data_dict


class SegDataset(BaseTaskDataset):
    def __getitem__(self, idx):
        file = self.files[idx]

        data = torch.load(file, map_location="cpu", weights_only=False)

        if not isinstance(data, torch.Tensor):
            raise RuntimeError(
                f"Expected {file} to contain a tensor, but found "
                f"{type(data).__name__}."
            )

        if data.ndim != 4:
            raise RuntimeError(
                f"Expected packed segmentation data with shape "
                f"[C + 1, H, W, D], but {file} produced "
                f"{tuple(data.shape)}."
            )

        if data.shape[0] < 2:
            raise RuntimeError(
                f"Expected at least one image channel and one label "
                f"channel in {file}, but found {data.shape[0]} channels."
            )

        image = _ensure_image_shape(data[:-1], file)
        label = _ensure_label_shape(data[-1:], file)

        if image.shape[1:] != label.shape[1:]:
            raise RuntimeError(
                f"Image and label spatial shapes do not match for "
                f"{file}: image={tuple(image.shape)}, "
                f"label={tuple(label.shape)}."
            )

        properties = load_pickle(
            file.replace(".pt", ".pkl")
        )

        data_dict = {
            "file_path": file,
            "image": image,
            "label": label,
            "foreground_locations": properties["foreground_locations"],
            "info": get_processed_data_info(file),
            "transforms_applied": {},
        }

        _validate_modality_count(data_dict, file)

        data_dict = self._transform(data_dict)

        # The task module accepts both names. Keep `label` for existing spatial
        # transforms and expose the explicit task key only after augmentation.
        data_dict["SEG_label"] = data_dict["label"]

        # foreground_locations is only required by transforms such as
        # foreground-aware cropping.
        data_dict.pop("foreground_locations", None)

        return data_dict


class ClsRegDataset(BaseTaskDataset):
    def __getitem__(self, idx):
        file = self.files[idx]

        data = torch.load(file, map_location="cpu", weights_only=False)

        if not isinstance(data, (tuple, list)):
            raise RuntimeError(
                f"Expected {file} to contain (image, label), but found "
                f"{type(data).__name__}."
            )

        if len(data) < 2:
            raise RuntimeError(
                f"Expected (image, label) in {file}, but found "
                f"{len(data)} elements."
            )

        image = _ensure_image_shape(data[0], file)
        label = torch.as_tensor(data[1])

        data_dict = {
            "file_path": file,
            "image": image,
            "CLSREG_label": label,
            "info": get_processed_data_info(file),
            "transforms_applied": {},
        }

        _validate_modality_count(data_dict, file)

        return self._transform(data_dict)


class SegTestDataset(BaseTaskDataset):
    def __getitem__(self, idx):
        file = self.files[idx]

        data = torch.load(file, map_location="cpu", weights_only=False)

        if not isinstance(data, torch.Tensor):
            raise RuntimeError(
                f"Expected {file} to contain a tensor, but found "
                f"{type(data).__name__}."
            )

        if data.ndim != 4 or data.shape[0] < 2:
            raise RuntimeError(
                f"Expected packed segmentation data with shape "
                f"[C + 1, H, W, D], but {file} produced "
                f"{tuple(data.shape)}."
            )

        image = _ensure_image_shape(data[:-1], file)
        label = _ensure_label_shape(data[-1:], file)

        if image.shape[1:] != label.shape[1:]:
            raise RuntimeError(
                f"Image and label spatial shapes do not match for "
                f"{file}: image={tuple(image.shape)}, "
                f"label={tuple(label.shape)}."
            )

        properties = load_pickle(file.replace(".pt", ".pkl"))

        src_label = self._get_src_label(file, properties)

        sample_id = "_".join(Path(file).parts[-3:]).replace(".pt", "")

        data_dict = {
            "file_path": file,
            "image": image,
            "label": label,
            "src_label": src_label,
            "properties": properties,
            "id": sample_id,
            "info": get_processed_data_info(file),
            "transforms_applied": {},
        }

        _validate_modality_count(data_dict, file)

        return self._transform(data_dict)

    def _get_src_label(
        self,
        file: str,
        properties: dict,
    ) -> torch.Tensor:
        # Original, unprocessed label used for restoring and evaluating
        # predictions in the source-image space.
        src_label_path = (
            file.replace(get_data_path(), get_source_labels_path(),).replace(".pt","_label.nii.gz")
        )

        src_label_nii = read_file_to_nifti_or_np(src_label_path)

        src_label_nii = reorient_nib_image(
            src_label_nii,
            original_orientation=properties[
                "original_orientation"
            ],
            target_orientation=properties["new_direction"],
        )

        src_label_npy = nifti_or_np_to_np(src_label_nii)

        # Preserve the existing [1, 1, H, W, D] convention used by
        # downstream test-time restoration/evaluation.
        return torch.from_numpy(src_label_npy).float().unsqueeze(0).unsqueeze(0)


class ClsRegTestDataset(BaseTaskDataset):
    def __getitem__(self, idx):
        file = self.files[idx]

        data = torch.load(file, map_location="cpu", weights_only=False)

        if not isinstance(data, (tuple, list)):
            raise RuntimeError(
                f"Expected {file} to contain (image, label), but found "
                f"{type(data).__name__}."
            )

        if len(data) < 2:
            raise RuntimeError(
                f"Expected (image, label) in {file}, but found "
                f"{len(data)} elements."
            )

        image = _ensure_image_shape(data[0], file)
        label = torch.as_tensor(data[1])

        data_dict = {
            "file_path": file,
            "image": image,
            "CLSREG_label": label,
            "info": get_processed_data_info(file),
            "transforms_applied": {},
        }

        _validate_modality_count(data_dict, file)

        return self._transform(data_dict)


class SingleSubjectPredictDataset(BaseTaskDataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(
                f"SingleSubjectPredictDataset only contains index 0, "
                f"but received index {idx}."
            )

        if not self.files:
            raise RuntimeError(
                "SingleSubjectPredictDataset received no files."
            )

        properties: dict[str, Any] = {}
        images = []
        spatial_shapes = set()

        reference_affine = None
        reference_spacing = None
        reference_direction = None

        for file in self.files:
            image, file_info = self._load_image(file)

            if image.shape[0] != 1:
                raise RuntimeError(
                    f"Expected one channel per input file, but {file} "
                    f"produced {image.shape[0]} channels."
                )

            spatial_shapes.add(tuple(image.shape[1:]))
            images.append(image)

            if file_info is not None:
                if reference_affine is None:
                    reference_affine = file_info["affine"]
                    reference_spacing = file_info["spacing"]
                    reference_direction = file_info["direction"]
                    properties.update(
                        file_info.get("properties", {})
                    )
                else:
                    self._validate_same_grid(
                        file=file,
                        file_info=file_info,
                        reference_affine=reference_affine,
                        reference_spacing=reference_spacing,
                        reference_direction=reference_direction,
                    )

        if len(spatial_shapes) != 1:
            raise RuntimeError(
                "Selected images do not have matching spatial shapes: "
                f"{self.files}"
            )

        image = torch.cat(images, dim=0)
        properties["original_size"] = tuple(image.shape[1:])

        if reference_affine is None:
            raise RuntimeError(
                "Could not determine affine, spacing, and direction. "
                "For .pt and .npy inputs, a matching .pkl metadata "
                "file is required."
            )

        modality_ids = torch.as_tensor(get_modality_id(self.files), dtype=torch.long).reshape(-1)

        data_dict = {
            "file_path": list(self.files),
            "image": image,
            "properties": properties,
            "info": {
                "affine": reference_affine,
                "spacing": reference_spacing,
                "direction": reference_direction,
                "modality": modality_ids,
            },
            "transforms_applied": {},
        }

        _validate_modality_count(
            data_dict,
            str(self.files),
        )

        return self._transform(data_dict)

    def _load_image(
        self,
        file: str,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        if file.endswith(".pt"):
            loaded = torch.load(file, map_location="cpu", weights_only=False)

            image = _ensure_image_shape(loaded, file)

            info = self._load_sidecar_info(file)
            return image, info

        if file.endswith(".npy"):
            image = torch.from_numpy(np.load(file))

            image = _ensure_image_shape(image, file)

            info = self._load_sidecar_info(file)
            return image, info

        if file.endswith((".nii", ".nii.gz")):
            nifti = nib.load(file)

            # Nibabel axes are retained as the project's [H, W, D] order.
            image = torch.from_numpy(
                np.asarray(nifti.dataobj, dtype=np.float32)
            ).unsqueeze(0)

            image = _ensure_image_shape(image, file)

            affine = torch.as_tensor(nifti.affine, dtype=torch.float32)

            info = {
                "affine": affine,
                "spacing": torch.as_tensor(nifti.header.get_zooms()[:3], dtype=torch.float32),
                "direction": "".join(aff2axcodes(nifti.affine)),
                "properties": {
                    "nifti_metadata": {
                        "affine": nifti.affine,
                        "header": nifti.header,
                        "reoriented": False,
                    },
                },
            }

            return image, info

        raise ValueError(
            f"Unsupported file type: {file}"
        )

    @staticmethod
    def _load_sidecar_info(
        file: str,
    ) -> Optional[dict]:
        if file.endswith(".pt"):
            sidecar = file.replace(".pt", ".pkl")
        elif file.endswith(".npy"):
            sidecar = file.replace(".npy", ".pkl")
        else:
            return None

        if not Path(sidecar).is_file():
            return None

        properties = load_pickle(sidecar)

        nifti_metadata = properties.get("nifti_metadata", {})

        affine = nifti_metadata.get("affine")
        spacing = properties.get("new_spacing")
        direction = properties.get("new_direction")

        if (
            affine is None
            or spacing is None
            or direction is None
        ):
            raise RuntimeError(
                f"Metadata file {sidecar} does not contain affine, "
                "new_spacing, and new_direction."
            )

        return {
            "affine": torch.as_tensor(affine, dtype=torch.float32),
            "spacing": torch.as_tensor(spacing, dtype=torch.float32),
            "direction": direction,
            "properties": properties,
        }

    @staticmethod
    def _validate_same_grid(
        file: str,
        file_info: dict,
        reference_affine: torch.Tensor,
        reference_spacing: torch.Tensor,
        reference_direction: str,
    ) -> None:
        if file_info["direction"] != reference_direction:
            raise RuntimeError(
                f"Direction mismatch for {file}: "
                f"{file_info['direction']} != {reference_direction}."
            )

        if not torch.allclose(
            file_info["spacing"],
            reference_spacing,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                f"Spacing mismatch for {file}: "
                f"{file_info['spacing'].tolist()} != "
                f"{reference_spacing.tolist()}."
            )

        if not torch.allclose(
            file_info["affine"],
            reference_affine,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RuntimeError(
                f"Affine mismatch for {file}."
            )


def full_volume_task_collate(samples: Sequence[Mapping[str, Any]]) -> dict:
    """Pad variable-size, variable-channel full volumes for one task batch.

    Images are padded only on the high end of each axis. The original valid
    shapes and real-channel mask are retained so the physical convolution and
    segmentation loss can ignore padding.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    images = [torch.as_tensor(sample["image"]).float() for sample in samples]
    if any(image.ndim != 4 for image in images):
        raise ValueError("Every image must have shape [C, H, W, D].")

    batch_size = len(images)
    max_channels = max(image.shape[0] for image in images)
    max_shape = tuple(max(image.shape[axis] for image in images) for axis in range(1, 4))
    image_batch = images[0].new_zeros((batch_size, max_channels, *max_shape))
    modality = torch.zeros((batch_size, max_channels), dtype=torch.long)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    spacing = torch.empty((batch_size, 3), dtype=torch.float32)
    valid_shapes = torch.empty((batch_size, 3), dtype=torch.long)

    for index, (sample, image) in enumerate(zip(samples, images)):
        channels, height, width, depth = image.shape
        image_batch[index, :channels, :height, :width, :depth] = image
        info = sample.get("info", {})

        sample_modality = torch.as_tensor(
            info.get("modality", torch.zeros(channels)), dtype=torch.long
        ).reshape(-1)
        if sample_modality.numel() != channels:
            raise ValueError(
                f"Sample {index} has {channels} channels but "
                f"{sample_modality.numel()} modality IDs."
            )
        modality[index, :channels] = sample_modality
        channel_mask[index, :channels] = True

        sample_spacing = torch.as_tensor(info["spacing"], dtype=torch.float32).reshape(-1)
        if sample_spacing.shape != (3,) or (sample_spacing <= 0).any():
            raise ValueError(f"Sample {index} has invalid spacing {sample_spacing}.")
        spacing[index] = sample_spacing

        sample_valid_shape = torch.as_tensor(
            info.get("valid_spatial_shapes", image.shape[1:]), dtype=torch.long
        ).reshape(-1)
        if sample_valid_shape.shape != (3,):
            raise ValueError("Each valid_spatial_shapes value must contain 3 entries.")
        actual_shape = torch.tensor(image.shape[1:], dtype=torch.long)
        valid_shapes[index] = torch.minimum(sample_valid_shape, actual_shape)

    batch: dict[str, Any] = {
        "image": image_batch,
        "file_path": [sample.get("file_path") for sample in samples],
        "info": {
            "spacing": spacing,
            "modality": modality,
            "channel_mask": channel_mask,
            "valid_spatial_shapes": valid_shapes,
            "affine": [sample.get("info", {}).get("affine") for sample in samples],
            "direction": [sample.get("info", {}).get("direction") for sample in samples],
        },
    }

    if all("CLSREG_label" in sample for sample in samples):
        labels = [torch.as_tensor(sample["CLSREG_label"]) for sample in samples]
        try:
            batch["CLSREG_label"] = torch.stack(labels)
        except RuntimeError as error:
            raise ValueError("CLSREG labels in a batch must have matching shapes.") from error

    segmentation_key = None
    if all("SEG_label" in sample for sample in samples):
        segmentation_key = "SEG_label"
    elif all("label" in sample for sample in samples):
        segmentation_key = "label"
    if segmentation_key is not None:
        labels = [_ensure_label_shape(sample[segmentation_key], str(index)) for index, sample in enumerate(samples)]
        max_label_channels = max(label.shape[0] for label in labels)
        label_batch = labels[0].new_zeros(
            (batch_size, max_label_channels, *max_shape)
        )
        for index, label in enumerate(labels):
            channels, height, width, depth = label.shape
            label_batch[index, :channels, :height, :width, :depth] = label
        batch["label"] = label_batch
        batch["SEG_label"] = label_batch

    # Keep non-tensor restoration metadata as per-sample lists.
    for key in ("id", "properties", "src_label", "transforms_applied"):
        if any(key in sample for sample in samples):
            batch[key] = [sample.get(key) for sample in samples]

    return batch
