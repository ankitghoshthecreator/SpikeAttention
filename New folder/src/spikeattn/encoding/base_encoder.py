"""Abstract base interface for continuous-to-spike signal encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseSpikeEncoder(ABC):
    """Abstract base class for converting continuous ECG waveforms into discrete spike rasters."""

    @abstractmethod
    def fit(self, X: np.ndarray | torch.Tensor) -> BaseSpikeEncoder:
        """Fit encoding parameters (e.g. threshold scale, baseline) on training data.

        Args:
            X: Continuous signals of shape [B, 2, L].

        Returns:
            self
        """
        pass

    @abstractmethod
    def encode(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Transform continuous signals into binary spike tensors.

        Args:
            X: Continuous signals of shape [B, 2, L].

        Returns:
            Spike tensor of shape [B, T, C] with values in {0.0, 1.0}.
        """
        pass
