"""Beat segment shape, alignment, and boundary padding tests."""

import numpy as np
import pytest

from spikeattn.preprocessing.segmentation import BeatSegmenter


def test_center_peak_and_boundary_shape(small_ecg):
    segments, offsets = BeatSegmenter(360).segment(small_ecg, np.array([500, 0]))
    assert segments.shape == (2, 2, 360)
    assert offsets.tolist() == [180, 180]
    assert np.array_equal(segments[0, :, 180], small_ecg[500])
    assert np.all(segments[1, :, :180] == 0)


def test_invalid_segmentation_inputs_raise(small_ecg):
    segmenter = BeatSegmenter(360)
    with pytest.raises(ValueError):
        segmenter.segment(small_ecg, np.array([-1]))
    with pytest.raises(ValueError):
        segmenter.segment(small_ecg, np.array([len(small_ecg) + 10]))
    with pytest.raises(ValueError):
        segmenter.segment(small_ecg[:, 0], np.array([100]))
