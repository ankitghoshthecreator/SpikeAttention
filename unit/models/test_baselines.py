"""Unit tests for baseline models: SVM, CNN, and RNN."""

import numpy as np
import pytest
import torch

from spikeattn.models.base_model import BaseModel
from spikeattn.models.baselines.cnn_baseline import CNNBaseline
from spikeattn.models.baselines.rnn_baseline import RNNBaseline
from spikeattn.models.baselines.svm_baseline import SVMBaseline


RNG = np.random.default_rng(123)
N_SAMPLES = 20
X_SIGNALS = RNG.normal(0, 1, (N_SAMPLES, 2, 360)).astype(np.float32)
Y_LABELS = RNG.integers(0, 5, size=N_SAMPLES).astype(np.int64)


class TestSVMBaseline:
    def test_implements_base_model(self):
        svm = SVMBaseline()
        assert isinstance(svm, BaseModel)
        assert svm.name == "svm"
        assert svm.count_parameters() == 0

    def test_fit_and_predict_contracts(self):
        svm = SVMBaseline()
        svm.fit(X_SIGNALS, Y_LABELS)

        probs = svm.predict_proba(X_SIGNALS)
        assert probs.shape == (N_SAMPLES, 5)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(N_SAMPLES), atol=1e-5)

        preds = svm.predict(X_SIGNALS)
        assert preds.shape == (N_SAMPLES,)
        assert set(preds).issubset({0, 1, 2, 3, 4})

    def test_predict_before_fit_raises(self):
        svm = SVMBaseline()
        with pytest.raises(RuntimeError):
            svm.predict_proba(X_SIGNALS)


class TestCNNBaseline:
    def test_implements_base_model(self):
        cnn = CNNBaseline()
        assert isinstance(cnn, BaseModel)
        assert cnn.name == "cnn"
        assert cnn.count_parameters() > 0

    def test_forward_logits_shape(self):
        cnn = CNNBaseline()
        logits = cnn.forward_logits(X_SIGNALS)
        assert logits.shape == (N_SAMPLES, 5)

    def test_fit_and_predict(self):
        cnn = CNNBaseline()
        cnn.fit(X_SIGNALS, Y_LABELS, epochs=1, batch_size=8)

        probs = cnn.predict_proba(X_SIGNALS)
        assert probs.shape == (N_SAMPLES, 5)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(N_SAMPLES), atol=1e-5)

        preds = cnn.predict(X_SIGNALS)
        assert preds.shape == (N_SAMPLES,)
        assert set(preds).issubset({0, 1, 2, 3, 4})


class TestRNNBaseline:
    def test_implements_base_model(self):
        rnn = RNNBaseline()
        assert isinstance(rnn, BaseModel)
        assert rnn.name == "rnn"
        assert rnn.count_parameters() > 0

    def test_forward_logits_shape(self):
        rnn = RNNBaseline()
        logits = rnn.forward_logits(X_SIGNALS)
        assert logits.shape == (N_SAMPLES, 5)

    def test_fit_and_predict(self):
        rnn = RNNBaseline()
        rnn.fit(X_SIGNALS, Y_LABELS, epochs=1, batch_size=8)

        probs = rnn.predict_proba(X_SIGNALS)
        assert probs.shape == (N_SAMPLES, 5)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(N_SAMPLES), atol=1e-5)

        preds = rnn.predict(X_SIGNALS)
        assert preds.shape == (N_SAMPLES,)
        assert set(preds).issubset({0, 1, 2, 3, 4})
