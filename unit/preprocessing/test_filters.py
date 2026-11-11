"""Filter response and shape checks against SciPy frequency response."""

import numpy as np
import pytest
from scipy.signal import sosfreqz

from spikeattn.preprocessing.filters import BandpassFilter


def test_filter_passband_and_shape(small_ecg):
    filt = BandpassFilter(360)
    filtered = filt.transform(small_ecg)
    assert filtered.shape == small_ecg.shape
    frequencies, response = sosfreqz(filt.sos, worN=4096, fs=360)
    gain = np.abs(response)
    assert gain[np.argmin(abs(frequencies - 10))] > 0.95
    assert gain[np.argmin(abs(frequencies - 0.1))] < 0.1
    assert gain[np.argmin(abs(frequencies - 80))] < 0.1


def test_invalid_filter_parameters_raise():
    with pytest.raises(ValueError):
        BandpassFilter(360, low_hz=0.0)
    with pytest.raises(ValueError):
        BandpassFilter(360, low_hz=50.0, high_hz=40.0)
    with pytest.raises(ValueError):
        BandpassFilter(360, low_hz=0.5, high_hz=200.0)  # > Nyquist


def test_too_short_signal_raises():
    filt = BandpassFilter(360)
    with pytest.raises(ValueError):
        filt.transform(np.zeros((5, 2)))
