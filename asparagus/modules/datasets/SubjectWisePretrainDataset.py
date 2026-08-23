import hashlib
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

import nibabel as nib
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from asparagus.functional.loading import (
    get_file_info,
    get_modality_id,
    load_image_file,
)


# Lower index means higher validation priority.
MODALITY_PRIORITY = [1, 5, 7]


class SubjectWisePretrainDataset(Dataset):
    """
    Each session contributes exactly one dataset index.

    Files within a session are clustered into compatible spatial groups using:

        - shape
        - affine
        - orientation
        - voxel spacing

    Training:
        - Select one compatible group uniformly.
        - Select up to `max_channels` files from that group randomly.
        - First select one file per modality.
        - Fill remaining positions with additional compatible runs.

    Validation:
        - Select the compatible group with the strongest coverage of
          MODALITY_PRIORITY.
        - Select files deterministically.
        - First select one file per modality according to priority.
        - Fill remaining positions deterministically.

    Example:

        Session:
            T1, T2      -> compatible group A
            FLAIR       -> compatible group B
            50 DWI      -> compatible group C

        Training:
            P(group A) = P(group B) = P(group C) = 1/3

            group A -> T1 + T2
            group B -> FLAIR
            group C -> up to max_channels random DWI runs

        Validation:
            group A is selected because T1 and T2 have higher priority.
    """

    CACHE_VERSION = 2

    def __init__(
        self,
        session_files: List[List[str]],
        transforms: Optional[torchvision.transforms.Compose] = None,
        max_channels: int = 4,
        modality_priority: Optional[List[int]] = None,
        is_validation: bool = False,
        affine_tolerance: float = 1e-3,
        spacing_tolerance: float = 1e-3,
        metadata_cache_path: Optional[str] = None,
        show_progress: bool = True,
    ):
        super().__init__()

        if max_channels < 1:
            raise ValueError(
                f"max_channels must be at least 1, got {max_channels}."
            )

        if modality_priority is None:
            modality_priority = MODALITY_PRIORITY

        modality_priority = [
            int(modality) for modality in modality_priority
        ]

        if len(modality_priority) != len(set(modality_priority)):
            raise ValueError(
                "modality_priority must not contain duplicate IDs."
            )

        self.session_files = session_files
        self.transforms = transforms
        self.max_channels = max_channels
        self.modality_priority = modality_priority
        self.is_validation = is_validation
        self.affine_tolerance = affine_tolerance
        self.spacing_tolerance = spacing_tolerance
        self.metadata_cache_path = metadata_cache_path
        self.show_progress = show_progress

        self.priority_rank = {
            modality: rank
            for rank, modality in enumerate(
                self.modality_priority
            )
        }

        self.session_records = self._load_or_build_index()

        if not self.session_records:
            raise RuntimeError(
                "No valid 3D images were found in the supplied sessions."
            )

    def __len__(self):
        # A session contributes exactly one dataset record.
        return len(self.session_records)

    def __getitem__(self, idx):
        session_record = self.session_records[idx]

        grid_group = self._select_grid_group(
            session_record["grid_groups"]
        )

        if self.is_validation:
            selected_records = (
                self._select_validation_records(grid_group)
            )
        else:
            selected_records = (
                self._select_training_records(grid_group)
            )

        selected_files = [
            record["file"] for record in selected_records
        ]

        modality_ids = torch.tensor(
            [record["modality"] for record in selected_records],
            dtype=torch.long,
        )

        images = []

        for file in selected_files:
            image = load_image_file(file)

            if image.ndim != 4:
                raise RuntimeError(
                    f"Expected [C, H, W, D], but {file} produced "
                    f"shape {tuple(image.shape)}."
                )

            if image.shape[0] != 1:
                raise RuntimeError(
                    f"Expected one channel per NIfTI file, but {file} "
                    f"produced {image.shape[0]} channels."
                )

            images.append(image)

        # Grid grouping should guarantee this, but retain a defensive check.
        spatial_shapes = {
            tuple(image.shape[1:]) for image in images
        }

        if len(spatial_shapes) != 1:
            raise RuntimeError(
                "Selected images do not have matching spatial shapes: "
                f"{selected_files}"
            )

        data = torch.cat(images, dim=0)

        if data.shape[0] > self.max_channels:
            raise RuntimeError(
                f"Loaded {data.shape[0]} channels, exceeding "
                f"max_channels={self.max_channels}."
            )

        info = dict(get_file_info(selected_files[0]))
        if "spacing" not in info:
            raise KeyError(f"get_file_info did not return spacing for {selected_files[0]}.")
        # Preserve get_file_info's axis convention because it should match
        # load_image_file; only normalize its representation and validate it.
        spacing = torch.as_tensor(info["spacing"], dtype=torch.float32).flatten()
        if spacing.numel() != 3:
            raise ValueError(
                f"Expected three spacing values, got {tuple(spacing.shape)}."
            )
        if not torch.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError(f"Invalid spacing for {selected_files[0]}: {spacing}.")
        info["spacing"] = spacing
        info["modality"] = modality_ids
        info["channel_mask"] = torch.ones(
            len(selected_records),
            dtype=torch.bool,
        )
        info["valid_spatial_shapes"] = torch.tensor(
            data.shape[-3:],
            dtype=torch.long,
        )

        data_dict = {
            "session_path": selected_files,
            "session_id": session_record["session_id"],
            "image": data,
            "transforms_applied": {},
            "info": info,
        }

        data_dict = self._transform(data_dict)

        # Transforms may replace ``image`` with global/local crop lists, so
        # sanitize the complete returned structure rather than only one key.
        data_dict = self._sanitize_tensor_tree(data_dict)

        return data_dict

    # ------------------------------------------------------------------
    # Compatible-grid selection
    # ------------------------------------------------------------------

    def _select_grid_group(
        self,
        grid_groups: List[List[dict]],
    ) -> List[dict]:
        """
        Training:
            Select grid groups uniformly. A group containing 50 files has
            the same probability as a group containing one file.

        Validation:
            Select the group with the strongest priority-modality coverage.
        """
        if len(grid_groups) == 1:
            return grid_groups[0]

        if not self.is_validation:
            group_index = torch.randint(
                low=0,
                high=len(grid_groups),
                size=(1,),
            ).item()

            return grid_groups[group_index]

        return max(
            grid_groups,
            key=self._validation_grid_score,
        )

    def _validation_grid_score(
        self,
        grid_group: List[dict],
    ) -> tuple:
        """
        Compare groups lexicographically according to priority presence.

        With priority [1, 2, 3]:

            T1 + T2 -> (1, 1, 0)
            T1      -> (1, 0, 0)
            FLAIR   -> (0, 0, 1)
            DWI     -> (0, 0, 0)

        Therefore, T1 + T2 is selected.
        """
        available_modalities = {
            record["modality"] for record in grid_group
        }

        priority_presence = tuple(
            int(modality in available_modalities)
            for modality in self.modality_priority
        )

        # After priority coverage, prefer greater modality diversity.
        modality_diversity = len(available_modalities)

        # Prefer fewer repeated acquisitions as the next tie-breaker.
        # This prevents a 50-run DWI group from winning merely by size.
        negative_group_size = -len(grid_group)

        # Final deterministic tie-breaker.
        first_file = min(
            record["file"] for record in grid_group
        )

        inverted_filename = tuple(
            -ord(character) for character in first_file
        )

        return (
            priority_presence,
            modality_diversity,
            negative_group_size,
            inverted_filename,
        )

    # ------------------------------------------------------------------
    # File selection within a compatible grid
    # ------------------------------------------------------------------

    def _select_training_records(
        self,
        grid_group: List[dict],
    ) -> List[dict]:
        """
        Randomly select up to max_channels files.

        One file per modality is selected first, preventing repeated runs
        from crowding out other modalities. Remaining channel positions are
        filled randomly from all unselected compatible runs.
        """
        if len(grid_group) <= self.max_channels:
            # Still shuffle the order during training.
            permutation = torch.randperm(len(grid_group))

            return [
                grid_group[index]
                for index in permutation.tolist()
            ]

        records_by_modality = self._group_by_modality(
            grid_group
        )

        modalities = list(records_by_modality)

        # If there are more modalities than available channel positions,
        # select modalities uniformly without replacement.
        modality_permutation = torch.randperm(
            len(modalities)
        ).tolist()

        modalities = [
            modalities[index]
            for index in modality_permutation
        ]

        selected_records = []
        selected_files = set()

        for modality in modalities:
            if len(selected_records) >= self.max_channels:
                break

            candidates = records_by_modality[modality]

            candidate_index = torch.randint(
                low=0,
                high=len(candidates),
                size=(1,),
            ).item()

            selected = candidates[candidate_index]

            selected_records.append(selected)
            selected_files.add(selected["file"])

        remaining_slots = (
            self.max_channels - len(selected_records)
        )

        if remaining_slots > 0:
            remaining_records = [
                record
                for record in grid_group
                if record["file"] not in selected_files
            ]

            if remaining_records:
                permutation = torch.randperm(
                    len(remaining_records)
                ).tolist()

                number_to_add = min(
                    remaining_slots,
                    len(remaining_records),
                )

                for index in permutation[:number_to_add]:
                    selected_records.append(
                        remaining_records[index]
                    )

        # Randomize channel ordering so the model cannot associate a channel
        # position with a fixed modality.
        if len(selected_records) > 1:
            permutation = torch.randperm(
                len(selected_records)
            ).tolist()

            selected_records = [
                selected_records[index]
                for index in permutation
            ]

        return selected_records

    def _select_validation_records(
        self,
        grid_group: List[dict],
    ) -> List[dict]:
        """
        Deterministically select up to max_channels files.

        Select one representative per modality first, ordered according to
        MODALITY_PRIORITY. Additional runs fill any remaining positions.
        """
        if len(grid_group) <= self.max_channels:
            return sorted(
                grid_group,
                key=self._validation_record_sort_key,
            )

        records_by_modality = self._group_by_modality(
            grid_group
        )

        modalities = sorted(
            records_by_modality,
            key=self._modality_sort_key,
        )

        selected_records = []
        selected_files = set()

        # First choose one deterministic representative per modality.
        for modality in modalities:
            if len(selected_records) >= self.max_channels:
                break

            candidates = sorted(
                records_by_modality[modality],
                key=lambda record: record["file"],
            )

            selected = candidates[0]

            selected_records.append(selected)
            selected_files.add(selected["file"])

        remaining_slots = (
            self.max_channels - len(selected_records)
        )

        if remaining_slots > 0:
            remaining_records = sorted(
                (
                    record
                    for record in grid_group
                    if record["file"] not in selected_files
                ),
                key=self._validation_record_sort_key,
            )

            selected_records.extend(
                remaining_records[:remaining_slots]
            )

        return selected_records

    @staticmethod
    def _group_by_modality(
        records: List[dict],
    ) -> dict:
        records_by_modality = defaultdict(list)

        for record in records:
            records_by_modality[
                record["modality"]
            ].append(record)

        return dict(records_by_modality)

    def _modality_sort_key(self, modality: int) -> tuple:
        """
        Priority modalities come first. Non-priority modality IDs follow in
        numeric order.
        """
        priority_rank = self.priority_rank.get(modality)

        if priority_rank is not None:
            return 0, priority_rank

        return 1, modality

    def _validation_record_sort_key(
        self,
        record: dict,
    ) -> tuple:
        return (
            self._modality_sort_key(record["modality"]),
            record["file"],
        )

    # ------------------------------------------------------------------
    # Cached metadata indexing
    # ------------------------------------------------------------------

    def _load_or_build_index(self) -> List[dict]:
        signature = self._dataset_signature()

        if self.metadata_cache_path is None:
            return self._build_index()

        cache_path = Path(self.metadata_cache_path)

        if cache_path.exists():
            try:
                tqdm.write(
                    f"Loading metadata cache: {cache_path}"
                )

                with cache_path.open("rb") as file:
                    cached = pickle.load(file)

                version_matches = (
                    cached.get("version")
                    == self.CACHE_VERSION
                )
                signature_matches = (
                    cached.get("signature")
                    == signature
                )
                session_records = cached.get(
                    "session_records"
                )

                if (
                    version_matches
                    and signature_matches
                    and session_records is not None
                ):
                    number_of_sessions = len(session_records)

                    number_of_groups = sum(
                        len(session["grid_groups"])
                        for session in session_records
                    )

                    number_of_images = sum(
                        len(group)
                        for session in session_records
                        for group in session["grid_groups"]
                    )

                    tqdm.write(
                        f"Loaded {number_of_sessions:,} sessions, "
                        f"{number_of_groups:,} compatible groups, and "
                        f"{number_of_images:,} image records."
                    )

                    return session_records

                if not version_matches:
                    reason = "cache version changed"
                elif not signature_matches:
                    reason = "input files or grouping settings changed"
                else:
                    reason = "session records are missing"

                tqdm.write(
                    f"Metadata cache is invalid because {reason}. "
                    "Rebuilding it."
                )

            except (
                EOFError,
                OSError,
                pickle.UnpicklingError,
                AttributeError,
                TypeError,
                ValueError,
            ) as error:
                tqdm.write(
                    f"Could not load metadata cache ({error}). "
                    "Rebuilding it."
                )

        session_records = self._build_index()

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_data = {
            "version": self.CACHE_VERSION,
            "signature": signature,
            "session_records": session_records,
        }

        temporary_path = cache_path.with_suffix(
            cache_path.suffix + f".{os.getpid()}.tmp"
        )

        tqdm.write(
            f"Saving metadata cache: {cache_path}"
        )

        try:
            with temporary_path.open("wb") as file:
                pickle.dump(
                    cache_data,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            os.replace(temporary_path, cache_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        tqdm.write(
            f"Saved metadata for {len(session_records):,} sessions."
        )

        return session_records

    def _build_index(self) -> List[dict]:
        session_records = []

        total_files = sum(
            len(session) for session in self.session_files
        )

        valid_3d_files = 0
        excluded_files = 0
        excluded_sessions = 0

        progress = tqdm(
            total=total_files,
            desc="Indexing NIfTI headers",
            unit="file",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )

        try:
            for session in self.session_files:
                if not session:
                    excluded_sessions += 1
                    continue

                modality_ids = torch.as_tensor(
                    get_modality_id(session)
                ).flatten()

                if len(modality_ids) != len(session):
                    raise RuntimeError(
                        "get_modality_id(session) must return one modality "
                        f"ID per file. Received {len(modality_ids)} IDs "
                        f"for {len(session)} files."
                    )

                records = []

                for file, modality_id in zip(
                    session,
                    modality_ids.tolist(),
                ):
                    try:
                        record = self._read_metadata(
                            file=file,
                            modality=int(modality_id),
                        )

                        if record is None:
                            excluded_files += 1
                        else:
                            records.append(record)
                            valid_3d_files += 1

                    finally:
                        progress.update(1)

                grid_groups = self._cluster_compatible_records(
                    records
                )

                if not grid_groups:
                    excluded_sessions += 1
                    continue

                session_records.append(
                    {
                        "session_id": self._session_identifier(
                            session
                        ),
                        "grid_groups": grid_groups,
                    }
                )

                progress.set_postfix(
                    sessions=len(session_records),
                    groups=sum(
                        len(item["grid_groups"])
                        for item in session_records
                    ),
                    valid=valid_3d_files,
                    excluded=excluded_files,
                )

        finally:
            progress.close()

        number_of_groups = sum(
            len(session["grid_groups"])
            for session in session_records
        )

        tqdm.write(
            f"Indexing completed: {len(session_records):,} sessions, "
            f"{number_of_groups:,} compatible groups, "
            f"{valid_3d_files:,} valid images, "
            f"{excluded_files:,} excluded images, and "
            f"{excluded_sessions:,} excluded sessions."
        )

        return session_records

    def _read_metadata(
        self,
        file: str,
        modality: int,
    ) -> Optional[dict]:
        try:
            image = nib.load(file)
        except Exception as error:
            raise RuntimeError(
                f"Could not read NIfTI header: {file}"
            ) from error

        # Exclude 4D sequences.
        if image.ndim != 3:
            return None

        return {
            "file": file,
            "modality": modality,
            "shape": tuple(
                int(value) for value in image.shape
            ),
            "spacing": tuple(
                float(value)
                for value in image.header.get_zooms()[:3]
            ),
            "orientation": tuple(
                nib.aff2axcodes(image.affine)
            ),
            "affine": np.asarray(
                image.affine,
                dtype=np.float32,
            ),
        }

    def _cluster_compatible_records(
        self,
        records: List[dict],
    ) -> List[List[dict]]:
        groups = []

        for record in records:
            matching_group = None

            for group in groups:
                if self._grids_match(
                    record,
                    group[0],
                ):
                    matching_group = group
                    break

            if matching_group is None:
                groups.append([record])
            else:
                matching_group.append(record)

        for group in groups:
            group.sort(
                key=self._validation_record_sort_key
            )

        # Stable ordering is necessary for deterministic validation.
        groups.sort(
            key=lambda group: min(
                record["file"] for record in group
            )
        )

        return groups

    def _grids_match(
        self,
        first: dict,
        second: dict,
    ) -> bool:
        return (
            # Required for torch.cat even when physical metadata matches.
            first["shape"] == second["shape"]

            and first["orientation"] == second["orientation"]

            and np.allclose(
                first["spacing"],
                second["spacing"],
                atol=self.spacing_tolerance,
                rtol=0.0,
            )

            and np.allclose(
                first["affine"],
                second["affine"],
                atol=self.affine_tolerance,
                rtol=0.0,
            )
        )

    @staticmethod
    def _session_identifier(
        session: List[str],
    ) -> str:
        if not session:
            return ""

        path = Path(session[0])
        parts = path.parts

        for index, part in enumerate(parts):
            if part.startswith("ses-"):
                return str(
                    Path(*parts[: index + 1])
                )

        return str(path.parent)

    def _dataset_signature(self) -> str:
        digest = hashlib.sha256()

        digest.update(
            str(self.CACHE_VERSION).encode("utf-8")
        )

        for session in self.session_files:
            for file in session:
                digest.update(
                    file.encode("utf-8")
                )
                digest.update(b"\0")

            digest.update(b"\n")

        # Only preprocessing/grouping arguments belong in the cache
        # signature. Training/validation selection does not change metadata.
        digest.update(
            str(self.affine_tolerance).encode("utf-8")
        )
        digest.update(
            str(self.spacing_tolerance).encode("utf-8")
        )

        return digest.hexdigest()

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        return data_dict

    @classmethod
    def _sanitize_tensor_tree(cls, value: Any) -> Any:
        """Replace non-finite floating values anywhere in a sample."""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all():
                return torch.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=4.0,
                    neginf=-1.0,
                )
            return value
        if isinstance(value, dict):
            return {
                key: cls._sanitize_tensor_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._sanitize_tensor_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._sanitize_tensor_tree(item) for item in value)
        return value
