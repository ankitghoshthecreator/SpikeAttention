"""Beat-centered ECG window extraction."""

import numpy as np


class BeatSegmenter:
    """Extract fixed-length windows centered on annotated R peaks."""

    def __init__(self, window_samples: int = 360):
        if window_samples < 2:
            raise ValueError("window_samples must be at least 2")
        self.window_samples = int(window_samples)
        self.r_peak_index = window_samples // 2

    def segment(self, signal: np.ndarray, annotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `[beats,channels,L]` segments and each peak's local index."""
        x = np.asarray(signal)
        peaks = np.asarray(annotations, dtype=np.int64)
        if x.ndim != 2 or peaks.ndim != 1 or np.any(peaks < 0) or np.any(peaks >= len(x)):
            raise ValueError("signal must be [samples,channels] and peaks valid sample indices")
        left = self.r_peak_index
        right = self.window_samples - left
        result = np.zeros((len(peaks), x.shape[1], self.window_samples), dtype=np.float32)
        for i, peak in enumerate(peaks):
            start, stop = peak - left, peak + right
            src_start, src_stop = max(0, start), min(len(x), stop)
            result[i, :, src_start - start:src_stop - start] = x[src_start:src_stop].T
        return result, np.full(len(peaks), self.r_peak_index, dtype=np.int64)
