"""Support Vector Machine (SVM) baseline model implementing BaseModel."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import torch

from spikeattn.models.base_model import BaseModel


class SVMBaseline(BaseModel):
    """Classical Support Vector Machine baseline for two-channel ECG heartbeat classification.

    Extracts statistical, amplitude, and morphological features across both leads,
    standardizes them, and predicts class probabilities via calibrated SVM decision functions.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 2000) -> None:
        self._name = "svm"
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        # Use SGDClassifier with modified Huber loss for calibrated probabilistic SVM
        self.clf = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-4,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.calibrated: CalibratedClassifierCV | None = None
        self.classes_ = np.arange(5, dtype=np.int64)
        self._fitted = False

    @property
    def name(self) -> str:
        return self._name

    def extract_features(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Extract multi-lead summary and temporal features from signals [B, 2, L]."""
        if isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X, dtype=np.float32)

        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError(f"SVMBaseline expects input of shape [B, 2, L], got {arr.shape}")

        B, _, L = arr.shape
        feats = []

        for b in range(B):
            sig0 = arr[b, 0, :]
            sig1 = arr[b, 1, :]

            # Lead 0 moments & statistics
            m0 = float(np.mean(sig0))
            s0 = float(np.std(sig0))
            min0 = float(np.min(sig0))
            max0 = float(np.max(sig0))
            energy0 = float(np.sum(sig0 ** 2)) / L

            # Lead 1 moments & statistics
            m1 = float(np.mean(sig1))
            s1 = float(np.std(sig1))
            min1 = float(np.min(sig1))
            max1 = float(np.max(sig1))
            energy1 = float(np.sum(sig1 ** 2)) / L

            # Inter-lead correlation
            corr = float(np.corrcoef(sig0, sig1)[0, 1]) if s0 > 1e-6 and s1 > 1e-6 else 0.0
            if not np.isfinite(corr):
                corr = 0.0

            # Sub-segment energy distribution (split into 4 temporal quadrants)
            quarters0 = [float(np.mean(q ** 2)) for q in np.array_split(sig0, 4)]
            quarters1 = [float(np.mean(q ** 2)) for q in np.array_split(sig1, 4)]

            row = [
                m0, s0, min0, max0, energy0,
                m1, s1, min1, max1, energy1,
                corr,
            ] + quarters0 + quarters1
            feats.append(row)

        return np.asarray(feats, dtype=np.float32)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        val_data: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor] | None = None,
    ) -> SVMBaseline:
        """Fit feature scaler and calibrated SVM classifier."""
        features = self.extract_features(X)
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=np.int64)

        X_scaled = self.scaler.fit_transform(features)

        # Handle class calibration
        self.clf.fit(X_scaled, y_arr)
        try:
            from sklearn.frozen import FrozenEstimator
            self.calibrated = CalibratedClassifierCV(FrozenEstimator(self.clf))
        except (ImportError, AttributeError):
            self.calibrated = CalibratedClassifierCV(self.clf, cv="prefit")
        self.calibrated.fit(X_scaled, y_arr)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return class probabilities of shape [B, 5]."""
        if not self._fitted:
            raise RuntimeError("SVMBaseline must be fitted before predict_proba()")

        features = self.extract_features(X)
        X_scaled = self.scaler.transform(features)
        probs_raw = self.calibrated.predict_proba(X_scaled)  # type: ignore[union-attr]

        # Ensure full 5-class distribution
        B = len(features)
        probs = np.zeros((B, 5), dtype=np.float64)
        for i, c in enumerate(self.calibrated.classes_):  # type: ignore[union-attr]
            if 0 <= c < 5:
                probs[:, c] = probs_raw[:, i]

        # Re-normalize rows to strictly sum to 1.0
        row_sums = probs.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return predicted class labels in {0, 1, 2, 3, 4}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1).astype(np.int64)

    def count_parameters(self) -> int:
        """Return 0: SVM has no gradient parameters.

        Use count_support_vectors() for the computed weight memory footprint.
        """
        return 0

    def count_support_vectors(self) -> int:
        """Return total number of support vectors across all calibrated sub-classifiers.

        Used by EfficiencyProfiler to compute weight memory as:
            n_support_vectors * n_features_in_ * 4 bytes (float32).
        Returns 0 if the model is not yet fitted.
        """
        if not self._fitted or self.calibrated is None:
            return 0
        total_sv = 0
        # CalibratedClassifierCV wraps one or more calibrated classifiers.
        # Each calibrated estimator exposes the base estimator via .estimator.
        try:
            for cal in self.calibrated.calibrated_classifiers_:
                estimator = getattr(cal, "estimator", None)
                if estimator is not None and hasattr(estimator, "coef_"):
                    total_sv += int(estimator.coef_.shape[1])  # SGDClassifier coef_: (n_classes, n_features)
        except AttributeError:
            pass
        return total_sv

    @property
    def n_features_in_(self) -> int:
        """Number of input features after extraction and scaling.

        Required by EfficiencyProfiler for weight memory calculation.
        """
        if not self._fitted or self.calibrated is None:
            return 0
        try:
            return int(self.scaler.n_features_in_)
        except AttributeError:
            return 0
