"""Unit tests for PlainSNN and SpikeAttentionSNN architectures."""

import numpy as np
import pytest
import torch

from spikeattn.models.base_model import BaseModel
from spikeattn.models.snn.plain_snn import PlainSNN
from spikeattn.models.snn.st_snn import SpikeAttentionSNN


RNG = np.random.default_rng(42)
N_SAMPLES = 8
X_CONT = RNG.normal(0, 1, (N_SAMPLES, 2, 360)).astype(np.float32)
X_SPIKES = RNG.choice([0.0, 1.0], size=(N_SAMPLES, 90, 4)).astype(np.float32)
Y_LABELS = RNG.integers(0, 5, size=N_SAMPLES).astype(np.int64)


class TestPlainSNN:
    def test_implements_base_model(self):
        model = PlainSNN()
        assert isinstance(model, BaseModel)
        assert model.name == "plain_snn"
        assert model.count_parameters() > 0

    def test_forward_with_continuous_input(self):
        model = PlainSNN()
        logits = model.forward_logits(X_CONT)
        assert logits.shape == (N_SAMPLES, 5)

    def test_forward_with_spike_input(self):
        model = PlainSNN()
        logits = model.forward_logits(X_SPIKES)
        assert logits.shape == (N_SAMPLES, 5)

    def test_predict_proba_and_predict(self):
        model = PlainSNN()
        probs = model.predict_proba(X_CONT)
        assert probs.shape == (N_SAMPLES, 5)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(N_SAMPLES), atol=1e-5)

        preds = model.predict(X_CONT)
        assert preds.shape == (N_SAMPLES,)
        assert set(preds).issubset({0, 1, 2, 3, 4})


class TestSpikeAttentionSNN:
    def test_implements_base_model(self):
        model = SpikeAttentionSNN()
        assert isinstance(model, BaseModel)
        assert model.name == "st_snn"
        assert model.count_parameters() > 0

    def test_forward_with_continuous_input(self):
        model = SpikeAttentionSNN()
        logits = model.forward_logits(X_CONT)
        assert logits.shape == (N_SAMPLES, 5)

    def test_forward_with_spike_input(self):
        model = SpikeAttentionSNN()
        logits = model.forward_logits(X_SPIKES)
        assert logits.shape == (N_SAMPLES, 5)

    def test_predict_proba_sum_to_one(self):
        model = SpikeAttentionSNN()
        probs = model.predict_proba(X_CONT)
        assert probs.shape == (N_SAMPLES, 5)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(N_SAMPLES), atol=1e-5)

    def test_predict_labels(self):
        model = SpikeAttentionSNN()
        preds = model.predict(X_CONT)
        assert preds.shape == (N_SAMPLES,)
        assert set(preds).issubset({0, 1, 2, 3, 4})

    def test_inter_lead_weights_recorded(self):
        model = SpikeAttentionSNN()
        model.forward_logits(X_CONT)
        assert model.last_lead_weights is not None
        assert model.last_lead_weights.shape == (N_SAMPLES, 2, 2)

    def test_reproducibility_fixed_seed(self):
        torch.manual_seed(999)
        m1 = SpikeAttentionSNN()
        out1 = m1.forward_logits(X_CONT)

        torch.manual_seed(999)
        m2 = SpikeAttentionSNN()
        out2 = m2.forward_logits(X_CONT)

        torch.testing.assert_close(out1, out2)
