"""Training-patient-only per-lead standardization."""

import numpy as np

from spikeattn.exceptions import InsufficientDataError, LeakageError


class LeadNormalizer:
    """Fit per-lead mean and standard deviation using training groups only."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_patient_group_ids: tuple[str, ...] = ()

    def fit(self, signals: np.ndarray, patient_group_ids: np.ndarray,
            train_groups: set[str] | tuple[str, ...] | None = None) -> "LeadNormalizer":
        """Fit moments only on explicitly named training patient groups.

        Signals may include other groups; those rows are excluded before moments
        are computed. An unknown training group is a leakage/integrity error.
        """
        x = np.asarray(signals, dtype=np.float64)
        groups = np.asarray(patient_group_ids).astype(str)
        if x.ndim != 3 or x.shape[1] != 2 or len(x) != len(groups) or not len(x):
            raise InsufficientDataError("Expected non-empty [N,2,L] signals and one group id per beat")
        if train_groups is None:
            raise LeakageError("train_groups must be explicit when fitting the normalizer")
        train_groups = set(map(str, train_groups))
        if not train_groups or not train_groups.issubset(set(groups.tolist())):
            raise LeakageError("Normalizer training groups are empty or absent from input data")
        selected = np.isin(groups, list(train_groups))
        x = x[selected]
        if not np.isfinite(x).all():
            raise ValueError("signals must be finite")
        self.mean_ = x.mean(axis=(0, 2))
        scale = x.std(axis=(0, 2))
        if np.any(scale <= np.finfo(float).eps):
            raise InsufficientDataError("A lead has zero variance in training data")
        self.scale_ = scale
        self.fit_patient_group_ids = tuple(sorted(train_groups))
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Standardize `[N,2,L]` using fitted training moments."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("LeadNormalizer.fit must be called before transform")
        x = np.asarray(signals, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("signals must have shape [N,2,L]")
        return ((x - self.mean_[None, :, None]) / self.scale_[None, :, None]).astype(np.float32)
