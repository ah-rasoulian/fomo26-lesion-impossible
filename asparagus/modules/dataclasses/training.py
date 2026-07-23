from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path


def group_by_session(files: list[str]) -> list[list[str]]:
    groups = defaultdict(list)

    for file in files:
        path = Path(file)
        parts = path.parts

        session_index = next(
            i for i, part in enumerate(parts)
            if part.startswith("ses-")
        )

        # Complete path through the session directory
        session_path = Path(*parts[: session_index + 1])
        groups[session_path].append(file)

    return list(groups.values())

@dataclass
class DataFiles:
    dataset_json: None
    splits: None
    test: None

@dataclass
class SubjectWiseDataFiles:
    def __init__(self, dataset_json=None, splits=None, test=None):
        self.dataset_json = dataset_json

        subject_wise_splits = {}
        for key, value in splits.items():
            subject_wise_splits[key] = group_by_session(value)
        self.splits = subject_wise_splits
        self.test = test