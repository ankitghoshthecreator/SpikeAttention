"""Test fixtures for isolated data/preprocessing checks."""

import numpy as np
import pytest


@pytest.fixture
def small_ecg():
    """Clearly synthetic signal used only for unit tests."""
    t = np.arange(1200) / 360
    return np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 7 * t)]).astype(np.float32)
