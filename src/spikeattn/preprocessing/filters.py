"""Zero-phase Butterworth bandpass filtering."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


class BandpassFilter:
    """Fourth-order Butterworth filter applied with forward-backward SOS."""

    def __init__(self, sampling_rate_hz: float, low_hz: float = 0.5, high_hz: float = 40.0, order: int = 4):
        if sampling_rate_hz <= 0 or low_hz <= 0 or high_hz <= low_hz or high_hz >= sampling_rate_hz / 2:
            raise ValueError("Require 0 < low_hz < high_hz < Nyquist and positive sampling rate")
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.order = int(order)
        self.sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz, output="sos")

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Filter `[samples, channels]` or one-dimensional signals."""
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim not in (1, 2) or not np.isfinite(values).all():
            raise ValueError("signal must be finite and one- or two-dimensional")
        if values.shape[0] < 3 * (2 * len(self.sos) + 1):
            raise ValueError("signal is too short for zero-phase filtering")
        return sosfiltfilt(self.sos, values, axis=0).astype(np.float32)
