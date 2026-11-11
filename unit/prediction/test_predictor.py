"""Unit tests for Predictor and PredictionBundle label resolution and contract enforcement."""

import numpy as np
import pandas as pd
import pytest
import torch

from spikeattn.models.base_model import BaseModel
from spikeattn.prediction.predictor import PredictionBundle, Predictor


class DummyModel(BaseModel):
    """Minimal dummy model implementing predict_proba."""

    name = "dummy_model"

    def __init__(self, n_classes=5):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(x), self.n_classes)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        n = len(X)
        p = np.full((n, self.n_classes), 1.0 / self.n_classes, dtype=np.float64)
        return p

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)

    def fit(self, X, y, val_data=None):
        return self

    def count_parameters(self) -> int:
        return 0


@pytest.fixture()
def sample_data():
    n = 10
    signals = np.random.randn(n, 2, 360).astype(np.float32)
    meta = {
        "beat_id": [f"b_{i}" for i in range(n)],
        "patient_group_id": ["mitdb:100"] * n,
        "record_id": ["100"] * n,
    }
    labels = np.random.randint(0, 5, size=n, dtype=np.int64)
    return signals, meta, labels, DummyModel()


class TestPredictor:
    def test_predict_with_canonical_y_true(self, sample_data):
        signals, meta, labels, model = sample_data
        meta["y_true"] = labels
        df_meta = pd.DataFrame(meta)

        bundle = Predictor.predict(model, signals, df_meta, seed=0, fold=0)
        assert isinstance(bundle, PredictionBundle)
        assert np.array_equal(bundle.y_true, labels)
        assert len(bundle.y_pred) == len(labels)
        assert bundle.probabilities.shape == (len(labels), 5)

    def test_predict_with_class_id_alias(self, sample_data):
        signals, meta, labels, model = sample_data
        meta["class_id"] = labels
        df_meta = pd.DataFrame(meta)

        bundle = Predictor.predict(model, signals, df_meta, seed=0, fold=0)
        assert np.array_equal(bundle.y_true, labels)

    def test_predict_with_label_alias(self, sample_data):
        signals, meta, labels, model = sample_data
        meta["label"] = labels
        df_meta = pd.DataFrame(meta)

        bundle = Predictor.predict(model, signals, df_meta, seed=0, fold=0)
        assert np.array_equal(bundle.y_true, labels)

    def test_predict_with_explicit_y_true(self, sample_data):
        signals, meta, labels, model = sample_data
        # meta without label columns
        df_meta = pd.DataFrame(meta)

        bundle = Predictor.predict(model, signals, df_meta, seed=0, fold=0, y_true=labels)
        assert np.array_equal(bundle.y_true, labels)

    def test_predict_explicit_y_true_mismatch_raises(self, sample_data):
        signals, meta, labels, model = sample_data
        meta["y_true"] = labels
        df_meta = pd.DataFrame(meta)

        conflicting_labels = (labels + 1) % 5
        with pytest.raises(ValueError, match="Explicit y_true does not match"):
            Predictor.predict(model, signals, df_meta, seed=0, fold=0, y_true=conflicting_labels)

    def test_predict_missing_ground_truth_raises_descriptive_error(self, sample_data):
        signals, meta, _, model = sample_data
        df_meta = pd.DataFrame(meta)  # No label, class_id, or y_true

        with pytest.raises(KeyError) as exc_info:
            Predictor.predict(model, signals, df_meta, seed=0, fold=0)

        err_msg = str(exc_info.value)
        assert "Ground-truth label column not found" in err_msg
        assert "Available fields" in err_msg

    def test_predict_missing_identifier_raises_descriptive_error(self, sample_data):
        signals, meta, labels, model = sample_data
        del meta["patient_group_id"]
        meta["y_true"] = labels
        df_meta = pd.DataFrame(meta)

        with pytest.raises(KeyError) as exc_info:
            Predictor.predict(model, signals, df_meta, seed=0, fold=0)

        err_msg = str(exc_info.value)
        assert "Required identifier 'patient_group_id' not found" in err_msg

    def test_predict_signals_and_metadata_length_mismatch_raises(self, sample_data):
        signals, meta, labels, model = sample_data
        meta["y_true"] = labels
        df_meta = pd.DataFrame(meta).iloc[:5]  # Only 5 rows, signals has 10

        with pytest.raises(ValueError, match="Row count mismatch"):
            Predictor.predict(model, signals, df_meta, seed=0, fold=0)

    def test_predict_invalid_class_indices_raises(self, sample_data):
        signals, meta, _, model = sample_data
        meta["y_true"] = [0, 1, 2, 99, 0, 1, 2, 3, 4, 0]  # 99 is invalid
        df_meta = pd.DataFrame(meta)

        with pytest.raises(ValueError, match="invalid class indices"):
            Predictor.predict(model, signals, df_meta, seed=0, fold=0)

    def test_prediction_bundle_length_validation(self):
        with pytest.raises(ValueError, match="PredictionBundle identifier length mismatch"):
            PredictionBundle(
                beat_ids=["b0", "b1"],
                patient_group_ids=["mitdb:100"],
                record_ids=["100", "100"],
                seed=0,
                fold=0,
                model_name="test",
                y_true=np.array([0, 1]),
                y_pred=np.array([0, 1]),
                probabilities=np.zeros((2, 5)),
            )
