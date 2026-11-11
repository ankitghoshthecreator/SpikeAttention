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


def _make_patient_df(model: str, seed: int = 0, fold: int = 0, n_patients: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        rows.append({
            "model": model,
            "seed": seed,
            "fold": fold,
            "patient_group_id": f"mitdb:{100 + i}",
            "n_beats": 100,
            "balanced_accuracy": 0.5 + i * 0.1,
            "accuracy": 0.6 + i * 0.1,
            "macro_f1": 0.4 + i * 0.1,
        })
    return pd.DataFrame(rows)


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

    def test_per_patient_multi_model_no_overwrite(self, store):
        """Critical regression test: saving metrics for 5 models (same seed/fold)
        must NOT overwrite earlier models' rows.  This was the root cause of the
        'baseline absent' bug where _append_or_write used per-column isin() logic
        instead of composite key matching.
        """
        model_names = ["svm", "cnn", "rnn", "plain_snn", "st_snn"]
        for m in model_names:
            store.save_per_patient_metrics(_make_patient_df(m, seed=0, fold=0))

        saved = pd.read_csv(store.metrics_dir / "per_patient.csv")
        # Each model has 2 patients — total must be 5 * 2 = 10 rows
        assert len(saved) == 10, (
            f"Expected 10 rows (5 models × 2 patients) but got {len(saved)}. "
            "Deduplication is incorrectly removing rows from earlier models."
        )
        # Every model must be present
        saved_models = set(saved["model"].unique())
        assert saved_models == set(model_names), (
            f"Missing models: {set(model_names) - saved_models}"
        )

    def test_per_patient_idempotent_on_rerun(self, store):
        """Saving the same model's metrics a second time must replace (not duplicate) them."""
        df = _make_patient_df("svm", seed=0, fold=0)
        store.save_per_patient_metrics(df)
        store.save_per_patient_metrics(df)  # second identical save

        saved = pd.read_csv(store.metrics_dir / "per_patient.csv")
        assert len(saved) == 2, (
            f"Expected 2 rows after idempotent save but got {len(saved)}"
        )

    def test_measurements_multi_model_no_overwrite(self, store):
        """Efficiency measurements for 5 models (same seed/fold) must all be retained."""
        for m in ["svm", "cnn", "rnn", "plain_snn", "st_snn"]:
            df = pd.DataFrame([{
                "model": m, "seed": 0, "fold": 0,
                "metric": "latency_ms", "value": 1.0,
                "unit": "ms", "kind": "measured",
                "method": "test", "device": "cpu",
            }])
            store.save_measurements(df)

        saved = pd.read_csv(store.efficiency_dir / "measurements.csv")
        assert len(saved) == 5, (
            f"Expected 5 measurement rows but got {len(saved)}"
        )
        assert set(saved["model"].unique()) == {"svm", "cnn", "rnn", "plain_snn", "st_snn"}
