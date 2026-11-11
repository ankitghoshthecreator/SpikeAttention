"""Unit tests for DeltaSpikeEncoder."""

import numpy as np
import pytest
import torch

from spikeattn.encoding.delta_encoder import DeltaSpikeEncoder


RNG = np.random.default_rng(42)
B, C, L = 4, 2, 360
SIGNALS_NP = RNG.normal(0, 1, (B, C, L)).astype(np.float32)
SIGNALS_TORCH = torch.from_numpy(SIGNALS_NP)


class TestDeltaSpikeEncoder:
    def test_fit_and_encode_numpy(self):
        encoder = DeltaSpikeEncoder(scale=0.5)
        encoder.fit(SIGNALS_NP)
        assert encoder.fitted is True
        assert encoder.threshold is not None
        assert encoder.threshold > 0.0

        spikes = encoder.encode(SIGNALS_NP)
        assert isinstance(spikes, torch.Tensor)
        assert spikes.shape == (B, L, 4)

    def test_encode_torch_tensor(self):
        encoder = DeltaSpikeEncoder(scale=0.5)
        encoder.fit(SIGNALS_TORCH)
        spikes = encoder.encode(SIGNALS_TORCH)
        assert spikes.shape == (B, L, 4)

    def test_spikes_are_strictly_binary(self):
        encoder = DeltaSpikeEncoder(scale=0.3)
        encoder.fit(SIGNALS_NP)
        spikes = encoder.encode(SIGNALS_NP)
        unique_vals = torch.unique(spikes).tolist()
        for v in unique_vals:
            assert v in {0.0, 1.0}

    def test_encode_before_fit_raises(self):
        encoder = DeltaSpikeEncoder()
        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.encode(SIGNALS_NP)

    def test_invalid_shape_raises(self):
        encoder = DeltaSpikeEncoder(threshold=0.1)
        with pytest.raises(ValueError):
            encoder.encode(np.zeros((B, 1, L), dtype=np.float32))

    def test_downsampling_target_len(self):
        encoder = DeltaSpikeEncoder(threshold=0.1, target_len=180)
        spikes = encoder.encode(SIGNALS_NP)
        assert spikes.shape == (B, 180, 4)
