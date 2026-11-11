"""Deterministic patient-group inner splitting and leakage assertions."""

from dataclasses import dataclass

import numpy as np
import re
from sklearn.model_selection import StratifiedGroupKFold

from spikeattn.exceptions import InsufficientDataError, LeakageError
from spikeattn.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class PatientPartition:
    """Immutable patient group sets for one outer fold."""
    seed: int
    fold: int
    train_patient_group_ids: tuple[str, ...]
    val_patient_group_ids: tuple[str, ...]
    test_patient_group_ids: tuple[str, ...]


def assert_disjoint(*partitions) -> None:
    """Raise when any patient group is repeated between partitions."""
    sets = [set(map(str, p)) for p in partitions]
    for i, left in enumerate(sets):
        for right in sets[i + 1:]:
            overlap = left & right
            if overlap:
                raise LeakageError(f"Patient groups occur in multiple partitions: {sorted(overlap)}")


class PatientSplitter:
    """Create an inner train/validation partition from outer-training groups."""

    @staticmethod
    def inner_split(groups, class_profiles: dict[str, dict[str, int]], seed: int,
                    validation_fraction: float = 0.2) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split groups deterministically, stratifying on each group's dominant class."""
        groups = sorted(set(map(str, groups)))
        if len(groups) < 2 or not 0 < validation_fraction < 1:
            raise InsufficientDataError("Need at least two groups and validation_fraction in (0,1)")
        labels = []
        for group in groups:
            profile = class_profiles.get(group, {})
            labels.append(max(profile, key=profile.get) if profile and max(profile.values()) > 0 else "_empty")
        # Rare dominant classes cannot be stratified; deterministic shuffle still keeps group isolation.
        counts = {label: labels.count(label) for label in set(labels)}
        inner_folds = max(2, round(1 / validation_fraction))
        if len(groups) >= inner_folds and min(counts.values()) >= inner_folds:
            splitter = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            indices = np.arange(len(groups))
            train_idx, val_idx = next(splitter.split(indices, labels, groups=groups))
        else:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(groups))
            n_val = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
            val_idx, train_idx = order[:n_val], order[n_val:]
        train, val = tuple(sorted(groups[i] for i in train_idx)), tuple(sorted(groups[i] for i in val_idx))
        assert_disjoint(train, val)
        return train, val


class PatientGroupCrossValidator:
    """Repeated stratified grouped folds with nested validation patients."""

    def __init__(self, n_splits: int = 5, seeds=(0, 1, 2, 3, 4), validation_fraction: float = 0.2):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.seeds = tuple(int(x) for x in seeds)
        self.validation_fraction = float(validation_fraction)

    def split(self, class_profiles: dict[str, dict[str, int]], dataset_hash: str = ""):
        """Yield (partition, stratification metadata) for all configured seeds/folds."""
        groups = sorted(map(str, class_profiles))
        if len(groups) < self.n_splits:
            raise InsufficientDataError(f"{len(groups)} groups cannot form {self.n_splits} folds")
        classes = ["N", "S", "V", "F", "Q"]
        labels = [max(class_profiles[g], key=class_profiles[g].get) if class_profiles[g] else "_empty" for g in groups]
        for seed in self.seeds:
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            indices = np.arange(len(groups))
            for fold, (train_outer, test_idx) in enumerate(cv.split(indices, labels, groups=groups)):
                outer_groups = [groups[i] for i in train_outer]
                train, val = PatientSplitter.inner_split(outer_groups, class_profiles, seed + fold,
                                                         self.validation_fraction)
                test = tuple(sorted(groups[i] for i in test_idx))
                assert_disjoint(train, val, test)
                part = PatientPartition(seed, fold, train, val, test)
                profiles = {g: {c: int(class_profiles[g].get(c, 0)) for c in classes} for g in groups}
                yield part, {"schema_version": "1.0.0", "seed": seed, "fold": fold,
                             "n_splits": self.n_splits, "dataset_hash": dataset_hash,
                             "train_patient_group_ids": list(train), "val_patient_group_ids": list(val),
                             "test_patient_group_ids": list(test),
                             "stratification": {"method": "StratifiedGroupKFold", "class_order": classes,
                                                "group_class_profiles": profiles,
                                                "balance_note": "Group-level dominant-class stratification; inspect per-fold class balance."}}

    def write(self, class_profiles: dict[str, dict[str, int]], output_dir, dataset_hash: str = "") -> list:
        """Write each split JSON using the frozen Split v1 filename."""
        import json
        from pathlib import Path
        target = Path(output_dir)
        if not re.fullmatch(r"[a-f0-9]{8,64}", dataset_hash):
            raise ValueError("dataset_hash must be a lowercase SHA-256 prefix of 8 to 64 hex characters")
        target.mkdir(parents=True, exist_ok=True)
        paths = []
        for partition, payload in self.split(class_profiles, dataset_hash):
            path = target / f"seed_{partition.seed}_fold_{partition.fold}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            paths.append(path)
        return paths
