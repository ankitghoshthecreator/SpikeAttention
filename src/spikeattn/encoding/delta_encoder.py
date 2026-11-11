"""Temporal threshold delta modulation encoding continuous 2-lead ECG into 4-channel spike rasters."""

from __future__ import annotations

import numpy as np
import torch

from spikeattn.encoding.base_encoder import BaseSpikeEncoder


class DeltaSpikeEncoder(BaseSpikeEncoder):
    """Encodes 2-channel continuous ECG signals [B, 2, L] into 4-channel spike rasters [B, T, 4].

    Channels are ordered as:
        Channel 0: Lead 0 ON spikes (positive delta >= +threshold)
        Channel 1: Lead 0 OFF spikes (negative delta <= -threshold)
        Channel 2: Lead 1 ON spikes (positive delta >= +threshold)
        Channel 3: Lead 1 OFF spikes (negative delta <= -threshold)

    Attributes:
        threshold: Absolute voltage threshold delta (or computed from scale * std).
        scale: Multiplier for standard deviation if threshold is fitted.
        fitted: Whether the encoder has been fitted.
    """

    def __init__(
        self,
        threshold: float | None = None,
        scale: float = 0.5,
        target_len: int | None = None,
    ) -> None:
        self.threshold = float(threshold) if threshold is not None else None
        self.scale = float(scale)
        self.target_len = target_len
        self.fitted = self.threshold is not None

    def fit(self, X: np.ndarray | torch.Tensor) -> DeltaSpikeEncoder:
        """Fit the delta threshold on training data using median absolute deviation or std.

        Args:
            X: Training signals of shape [B, 2, L].

        Returns:
            self
        """
        if isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X, dtype=np.float32)

        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError(f"Expected input of shape [B, 2, L], got {arr.shape}")

        # Compute average standard deviation across both leads
        stds = np.std(arr, axis=-1)
        mean_std = float(np.mean(stds))
        self.threshold = max(1e-4, self.scale * mean_std)
        self.fitted = True
        return self

    def encode(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode continuous signals into binary spike tensor [B, T, 4].

        Args:
            X: Input signals of shape [B, 2, L].

        Returns:
            Binary spike tensor of shape [B, T, 4] with values in {0.0, 1.0}.
        """
        if not self.fitted:
            raise RuntimeError("DeltaSpikeEncoder must be fitted before encode()")

        if isinstance(X, torch.Tensor):
            signals = X.detach().to(torch.float32)
        else:
            signals = torch.from_numpy(np.asarray(X, dtype=np.float32))

        if signals.ndim != 3 or signals.shape[1] != 2:
            raise ValueError(f"Expected signals of shape [B, 2, L], got {signals.shape}")

        B, C, L = signals.shape
        th = float(self.threshold)  # type: ignore[arg-type]

        # Downsample along time dimension if target_len is specified
        if self.target_len is not None and self.target_len < L:
            step = L // self.target_len
            signals = signals[:, :, : self.target_len * step : step]
            L = signals.shape[-1]

        # Fast vectorized delta modulation across batches and leads simultaneously
        # Lead 0: ON (ch 0), OFF (ch 1); Lead 1: ON (ch 2), OFF (ch 3) -> Total 4 channels
        spikes = torch.zeros((B, L, 4), dtype=torch.float32, device=signals.device)
        ref = signals[:, :, 0]  # [B, 2]

        for t in range(1, L):
            val = signals[:, :, t]  # [B, 2]
            diff = val - ref
            on = diff >= th
            off = diff <= -th

            spikes[:, t, 0] = on[:, 0].to(torch.float32)
            spikes[:, t, 1] = off[:, 0].to(torch.float32)
            spikes[:, t, 2] = on[:, 1].to(torch.float32)
            spikes[:, t, 3] = off[:, 1].to(torch.float32)

            any_spike = on | off
            ref = torch.where(any_spike, val, ref)

        return spikes
