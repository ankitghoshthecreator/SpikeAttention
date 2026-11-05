"""Unit tests for efficiency_analysis.py: validation, summarize, compare, energy_estimate."""

import numpy as np
import pandas as pd
import pytest

from spikeattn.statistics.efficiency_analysis import EfficiencyAnalyzer


def _make_efficiency_df(kinds=None):
    """Create a minimal valid efficiency measurements DataFrame."""
    if kinds is None:
        kinds = ["measured", "computed", "estimated"]
    rows = []
    for model in ["st_snn", "plain_snn"]:
        for metric, unit, kind in zip(
            ["latency_ms", "params", "energy_uJ"],
            ["ms", "count", "uJ"],
            kinds,
        ):
            for _ in range(5):
                rows.append({
                    "model": model,
                    "metric": metric,
                    "value": np.random.default_rng(42).uniform(1, 100),
                    "unit": unit,
                    "kind": kind,
                })
    return pd.DataFrame(rows)


class TestValidateMeasurementsFrame:
    def test_valid_df_passes(self):
        df = _make_efficiency_df()
        EfficiencyAnalyzer.validate_measurements_frame(df)  # must not raise

    def test_invalid_kind_raises(self):
        df = _make_efficiency_df()
        df.loc[0, "kind"] = "guessed"  # invalid
        with pytest.raises(ValueError, match="Invalid efficiency measurement kind"):
            EfficiencyAnalyzer.validate_measurements_frame(df)

    def test_missing_column_raises(self):
        df = _make_efficiency_df().drop(columns=["kind"])
        with pytest.raises(ValueError, match="missing required columns"):
            EfficiencyAnalyzer.validate_measurements_frame(df)


class TestSummarize:
    def test_returns_dataframe(self):
        df = _make_efficiency_df()
        summary = EfficiencyAnalyzer.summarize(df)
        assert isinstance(summary, pd.DataFrame)

    def test_has_model_column(self):
        df = _make_efficiency_df()
        summary = EfficiencyAnalyzer.summarize(df)
        assert "model" in summary.columns

    def test_has_median_column(self):
        df = _make_efficiency_df()
        summary = EfficiencyAnalyzer.summarize(df)
        assert "median" in summary.columns


class TestCompare:
    def test_returns_dataframe_with_relative_diff(self):
        df = _make_efficiency_df()
        comp = EfficiencyAnalyzer.compare(df, baseline="plain_snn")
        assert isinstance(comp, pd.DataFrame)
        assert "relative_diff_pct" in comp.columns

    def test_baseline_relative_diff_is_zero(self):
        df = _make_efficiency_df()
        comp = EfficiencyAnalyzer.compare(df, baseline="plain_snn")
        base_rows = comp[comp["model"] == "plain_snn"]
        for _, row in base_rows.iterrows():
            if row["relative_diff_pct"] is not None:
                assert row["relative_diff_pct"] == pytest.approx(0.0, abs=1e-10)


class TestEnergyEstimate:
    def test_kind_is_estimated(self):
        result = EfficiencyAnalyzer.energy_estimate(macs=1_000_000, synops=500_000)
        assert result["kind"] == "estimated"

    def test_total_equals_sum(self):
        macs, synops = 1_000_000, 500_000
        e_mac, e_ac = 4.6e-12, 0.9e-12
        result = EfficiencyAnalyzer.energy_estimate(macs, synops, e_mac=e_mac, e_ac=e_ac)
        expected_total = macs * e_mac + synops * e_ac
        assert result["total_energy_joules"] == pytest.approx(expected_total, rel=1e-10)

    def test_unit_is_joules(self):
        result = EfficiencyAnalyzer.energy_estimate(macs=1000, synops=1000)
        assert result["unit"] == "joules"

    def test_contains_assumptions(self):
        result = EfficiencyAnalyzer.energy_estimate(macs=1000, synops=1000)
        assert "assumptions" in result
        assert "literature_source" in result["assumptions"]
