"""Abstract base model contract for all classical and neural ECG classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseModel(ABC):
    """Abstract base class establishing the Model Contract v1.0.0.

    All classical estimators (e.g. SVM) and neural network architectures (CNN, RNN,
    PlainSNN, SpikeAttentionSNN) adhere to this common interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier (e.g. 'st_snn', 'svm', 'cnn', 'rnn', 'plain_snn')."""
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        val_data: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor] | None = None,
    ) -> BaseModel:
        """Fit model on training set with optional validation data for early stopping.

        Args:
            X: Input signals of shape [B, 2, 360] or spike trains [B, T, 4].
            y: Integer target class labels of shape [B] in {0, 1, 2, 3, 4}.
            val_data: Optional tuple (X_val, y_val) for validation monitoring.

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return class probabilities of shape [B, 5] summing to 1.0 in order (N, S, V, F, Q).

        Args:
            X: Input signals of shape [B, 2, 360] or spike trains [B, T, 4].

        Returns:
            Array of shape [B, 5] with values in [0, 1] summing to 1 along axis 1.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return class index predictions of shape [B] in {0, 1, 2, 3, 4}.

        Args:
            X: Input signals of shape [B, 2, 360] or spike trains [B, T, 4].

        Returns:
            Integer array of shape [B].
        """
        pass

    @abstractmethod
    def count_parameters(self) -> int:
        """Return the number of trainable model parameters (0 for classical models)."""
        pass
