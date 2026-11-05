"""Class-weighted cross-entropy loss for addressing extreme AAMI class imbalance."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ClassWeightedCrossEntropy(nn.Module):
    """Cross-Entropy Loss with balanced class weighting: w_c = N / (K * N_c).

    Ensures minority classes (S, V, F, Q) receive proportional gradient signal
    relative to the predominant normal (N) rhythm class.
    """

    def __init__(
        self,
        class_counts: dict[int, int] | np.ndarray | torch.Tensor | None = None,
        n_classes: int = 5,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        if class_counts is not None:
            weights = self.compute_weights(class_counts, n_classes, smoothing)
            self.register_buffer("weights", weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)
        else:
            self.register_buffer("weights", None)
            self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def compute_weights(
        counts: dict[int, int] | np.ndarray | torch.Tensor,
        n_classes: int = 5,
        smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Compute balanced weights w_c = total / (K * count_c)."""
        if isinstance(counts, dict):
            arr = np.array([counts.get(c, 0) for c in range(n_classes)], dtype=np.float32)
        elif isinstance(counts, torch.Tensor):
            arr = counts.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(counts, dtype=np.float32)

        if len(arr) < n_classes:
            padded = np.zeros(n_classes, dtype=np.float32)
            padded[: len(arr)] = arr
            arr = padded

        total = np.sum(arr)
        weights = np.zeros(n_classes, dtype=np.float32)

        for c in range(n_classes):
            cnt = arr[c] + smoothing
            if cnt > 0:
                weights[c] = total / (n_classes * cnt)
            else:
                weights[c] = 1.0

        # Normalize weights so mean is 1.0
        mean_w = np.mean(weights)
        if mean_w > 0:
            weights /= mean_w

        return torch.from_numpy(weights).float()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Evaluate weighted cross-entropy loss."""
        return self.criterion(logits, target)
