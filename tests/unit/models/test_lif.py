"""Unit tests for LIFNode spiking neuron."""

import numpy as np
import pytest
import torch

from spikeattn.models.snn.lif import LIFNode


class TestLIFNode:
    def test_euler_step_response_decay(self):
        """Under zero input current, membrane potential decays according to beta = 1 - dt/tau."""
        tau, dt = 10.0, 1.0
        lif = LIFNode(tau=tau, dt=dt, v_th=10.0, v_reset=0.0)  # High threshold to prevent firing

        # Initial subthreshold potential
        v0 = 0.8
        decay_factor = 1.0 - (dt / tau)

        # Feed single step of zero current with v0 manually set
        lif.v_mem = torch.tensor([[v0]])
        zero_input = torch.tensor([[0.0]])
        s, v = lif(zero_input)

        assert s.item() == 0.0
        expected_v = decay_factor * v0
        assert v.item() == pytest.approx(expected_v, rel=1e-5)

    def test_firing_and_hard_reset(self):
        """Input exceeding threshold fires a spike (S=1) and resets membrane potential."""
        lif = LIFNode(tau=10.0, dt=1.0, v_th=1.0, v_reset=0.0)

        # Large input current to trigger spike
        strong_input = torch.tensor([[2.0]])
        s, v = lif(strong_input)

        assert s.item() == 1.0
        assert v.item() == pytest.approx(0.0)  # Hard reset to v_reset

    def test_3d_sequence_processing(self):
        B, T, C = 2, 20, 8
        lif = LIFNode(tau=10.0, dt=1.0, v_th=1.0, v_reset=0.0)
        x = torch.randn(B, T, C)

        spikes, voltages = lif(x)
        assert spikes.shape == (B, T, C)
        assert voltages.shape == (B, T, C)

        # Spikes must be in {0.0, 1.0}
        unique_spikes = torch.unique(spikes).tolist()
        for s in unique_spikes:
            assert s in {0.0, 1.0}

    def test_reset_membrane_potential(self):
        lif = LIFNode()
        lif(torch.tensor([[0.5]]))
        assert lif.v_mem is not None
        lif.reset_membrane_potential()
        assert lif.v_mem is None
