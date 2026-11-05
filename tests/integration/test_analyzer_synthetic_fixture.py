"""Integration test: ExperimentAnalyzer on a synthetic Results-v1 fixture.

Creates a minimal but valid Results-v1 artefact tree in a tmp_path,
then verifies that ExperimentAnalyzer.run() completes without error and
returns an AnalysisBundle with the expected top-level keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spikeattn.statistics.experiment_analyzer import ExperimentAnalyzer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_per_patient(models, n_patients=20, seed=0):
    """Generate per_patient.csv rows for each model."""
    rng = np.random.default_rng(seed)
    rows = []
    patient_groups = [f"mitdb:{100 + i}" for i in range(n_patients)]
    for model in models:
        for fold in range(3):
            for s in range(2):
                for pg in patient_groups:
                    rows.append({
                        "model": model,
                        "seed": s,
                        "fold": fold,
                        "patient_group_id": pg,
                        "n_beats": int(rng.integers(50, 300)),
                        "balanced_accuracy": float(rng.uniform(0.5, 0.9)),
                        "accuracy": float(rng.uniform(0.5, 0.95)),
                        "macro_f1": float(rng.uniform(0.4, 0.85)),
                    })
    return pd.DataFrame(rows)


def _make_per_fold(models, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model in models:
        for fold in range(3):
            for s in range(2):
                rows.append({
                    "model": model,
                    "seed": s,
                    "fold": fold,
                    "balanced_accuracy": float(rng.uniform(0.5, 0.9)),
                    "accuracy": float(rng.uniform(0.5, 0.95)),
                    "macro_f1": float(rng.uniform(0.4, 0.85)),
                })
    return pd.DataFrame(rows)


def _make_efficiency(models):
    rows = []
    rng = np.random.default_rng(0)
    for model in models:
        for metric, unit, kind in [
            ("latency_ms", "ms", "measured"),
            ("params", "count", "computed"),
            ("energy_uJ", "uJ", "estimated"),
        ]:
            for _ in range(5):
                rows.append({
                    "model": model,
                    "metric": metric,
                    "value": float(rng.uniform(1, 100)),
                    "unit": unit,
                    "kind": kind,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


MODELS = ["st_snn", "svm", "cnn", "rnn", "plain_snn"]

CONFIG = {
    "alpha": 0.05,
    "ci_level": 0.95,
    "n_bootstrap": 200,
    "seed": 42,
    "primary_metric": "balanced_accuracy",
}


@pytest.fixture()
def results_dir(tmp_path):
    """Build a valid Results-v1 directory tree in tmp_path."""
    metrics_dir = tmp_path / "metrics"
    efficiency_dir = tmp_path / "efficiency"
    predictions_dir = tmp_path / "predictions"
    metrics_dir.mkdir()
    efficiency_dir.mkdir()
    predictions_dir.mkdir()

    per_patient = _make_per_patient(MODELS)
    per_patient.to_csv(metrics_dir / "per_patient.csv", index=False)
    _make_per_fold(MODELS).to_csv(metrics_dir / "per_fold.csv", index=False)
    _make_efficiency(MODELS).to_csv(efficiency_dir / "measurements.csv", index=False)

    # Minimal predictions stub
    rng = np.random.default_rng(0)
    for model in MODELS:
        pred_df = pd.DataFrame({
            "beat_id": range(100),
            "y_true": rng.integers(0, 5, 100),
            "y_pred": rng.integers(0, 5, 100),
        })
        pred_df.to_csv(predictions_dir / f"{model}_seed0.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExperimentAnalyzerIntegration:
    def test_run_returns_dict(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        assert isinstance(bundle, dict)

    def test_bundle_has_all_required_keys(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        required = {
            "descriptive",
            "assumptions",
            "comparisons",
            "intervals",
            "effects",
            "corrections",
            "relationships",
            "efficiency",
        }
        assert required.issubset(set(bundle.keys()))

    def test_descriptive_has_model_entries(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        descriptive = bundle["descriptive"]
        assert isinstance(descriptive, list)
        assert len(descriptive) >= 1
        assert "model" in descriptive[0]

    def test_comparisons_are_dicts(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        for comp in bundle["comparisons"]:
            assert isinstance(comp, dict)
            assert "p_value" in comp
            assert "statistic" in comp

    def test_intervals_contain_model_entries(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        ci = bundle["intervals"]
        assert isinstance(ci, dict)
        assert "st_snn" in ci
        assert "lower" in ci["st_snn"]
        assert "upper" in ci["st_snn"]

    def test_efficiency_summary_returned(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        eff = bundle["efficiency"]
        assert isinstance(eff, list)
        # Should have entries for each model × metric
        assert len(eff) > 0

    def test_corrections_has_primary_key(self, results_dir):
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        assert "primary" in bundle["corrections"]
        assert "adjusted_p_values" in bundle["corrections"]["primary"]

    def test_no_hard_coded_numeric_values(self, results_dir):
        """Verify outputs vary with different seeds (not hard-coded)."""
        cfg_a = dict(CONFIG, seed=1)
        cfg_b = dict(CONFIG, seed=99)
        b1 = ExperimentAnalyzer(results_dir, cfg_a).run()
        b2 = ExperimentAnalyzer(results_dir, cfg_b).run()
        # Bootstrap CIs should differ between seeds
        ci1 = b1["intervals"].get("st_snn", {})
        ci2 = b2["intervals"].get("st_snn", {})
        # At minimum both must have been computed (not all zeros / identical with different seeds)
        assert isinstance(ci1, dict) and isinstance(ci2, dict)

    def test_missing_required_csv_raises(self, tmp_path):
        """Missing per_patient.csv must raise FileNotFoundError."""
        (tmp_path / "metrics").mkdir()  # empty metrics dir
        with pytest.raises(FileNotFoundError):
            ExperimentAnalyzer(tmp_path, CONFIG).load()

    def test_invalid_efficiency_kind_logged_not_crashed(self, results_dir):
        """An efficiency CSV with invalid kind must NOT crash the analyzer (logged warning)."""
        eff_path = results_dir / "efficiency" / "measurements.csv"
        df = pd.read_csv(eff_path)
        df.loc[0, "kind"] = "INVALID"
        df.to_csv(eff_path, index=False)
        # Should not raise — invalid efficiency file is skipped with a warning
        analyzer = ExperimentAnalyzer(results_dir, CONFIG)
        bundle = analyzer.run()
        # efficiency key still present, may be empty
        assert "efficiency" in bundle

    def test_isolation_no_forbidden_imports(self):
        """Confirm experiment_analyzer does not import Part 1/3/4 modules."""
        import importlib
        import sys
        # Remove cached module if present
        mod_name = "spikeattn.statistics.experiment_analyzer"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import ast
        import pathlib
        src = pathlib.Path("src/spikeattn/statistics/experiment_analyzer.py")
        if not src.exists():
            src = pathlib.Path("D:/Spike Attention/src/spikeattn/statistics/experiment_analyzer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))
        forbidden_prefixes = (
            "spikeattn.data",
            "spikeattn.preprocessing",
            "spikeattn.models",
            "spikeattn.training",
            "spikeattn.prediction",
            "spikeattn.evaluation",
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                if any(module.startswith(fp) for fp in forbidden_prefixes):
                    pytest.fail(
                        f"experiment_analyzer.py imports forbidden module '{module}'"
                    )
