"""Unit tests for ResultsStore conformity to Results Contract v1.0.0."""

import numpy as np
import pandas as pd
import pytest

from spikeattn.evaluation.results_store import ResultsStore
from spikeattn.prediction.predictor import PredictionBundle


@pytest.fixture()
def store(tmp_path):
    return ResultsStore(tmp_path / "test_run")


def _make_sample_bundle(model="st_snn", seed=0, fold=0, n=5):
    probs = np.full((n, 5), 0.2, dtype=np.float64)
    return PredictionBundle(
        beat_ids=[f"beat_{i}" for i in range(n)],
        patient_group_ids=[f"mitdb:10{i}" for i in range(n)],
        record_ids=[f"10{i}" for i in range(n)],
        seed=seed,
        fold=fold,
        model_name=model,
        y_true=np.zeros(n, dtype=np.int64),
        y_pred=np.zeros(n, dtype=np.int64),
        probabilities=probs,
    )


class TestResultsStore:
    def test_save_predictions_schema(self, store):
        bundle = _make_sample_bundle()
        path = store.save_predictions(bundle)
        assert path.exists()

        df = pd.read_csv(path)
        required_cols = [
            "beat_id", "patient_group_id", "record_id", "seed", "fold",
            "model", "y_true", "y_pred", "p_N", "p_S", "p_V", "p_F", "p_Q"
        ]
        assert list(df.columns) == required_cols
        assert len(df) == 5

    def test_is_fold_completed(self, store):
        assert store.is_fold_completed("st_snn", seed=0, fold=0) is False
        bundle = _make_sample_bundle(model="st_snn", seed=0, fold=0)
        store.save_predictions(bundle)
        assert store.is_fold_completed("st_snn", seed=0, fold=0) is True
        assert store.is_fold_completed("st_snn", seed=0, fold=1) is False

    def test_save_per_patient_metrics_schema(self, store):
        df = pd.DataFrame([{
            "model": "st_snn",
            "seed": 0,
            "fold": 0,
            "patient_group_id": "mitdb:100",
            "n_beats": 100,
            "balanced_accuracy": 0.85,
            "accuracy": 0.90,
        }])
        path = store.save_per_patient_metrics(df)
        assert path.exists()
        saved = pd.read_csv(path)
        assert set(df.columns).issubset(set(saved.columns))

    def test_save_measurements_schema(self, store):
        df = pd.DataFrame([{
            "model": "st_snn",
            "seed": 0,
            "fold": 0,
            "metric": "latency_ms",
            "value": 1.25,
            "unit": "ms",
            "kind": "measured",
            "method": "warmup_median",
            "device": "cpu",
        }])
        path = store.save_measurements(df)
        assert path.exists()
        saved = pd.read_csv(path)
        assert "kind" in saved.columns
        assert saved["kind"].iloc[0] == "measured"
