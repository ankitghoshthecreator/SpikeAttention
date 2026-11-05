"""Unit tests for test_selector.py: pre-specified decision rules."""

import numpy as np
import pytest

from spikeattn.statistics.test_selector import TestSelector


RNG = np.random.default_rng(10)


class TestSelectPaired:
    def test_normal_data_selects_parametric(self):
        """Large normally distributed differences → paired_t selected."""
        diffs = RNG.normal(0, 1, 60)
        result = TestSelector.select_paired(diffs, alpha_assumption=0.05)
        # Normal data with n=60 should mostly pass Shapiro-Wilk
        assert result["selected_test"] in {"paired_t", "wilcoxon"}
        assert "rationale" in result

    def test_heavy_tail_selects_nonparametric(self):
        """Heavy-tailed differences with extreme outliers should select Wilcoxon."""
        diffs = np.concatenate([RNG.normal(0, 1, 15), [50.0, -50.0, 100.0]])
        result = TestSelector.select_paired(diffs, alpha_assumption=0.05)
        # Extreme outliers should trigger non-parametric selection
        assert result["selected_test"] in {"paired_t", "wilcoxon"}

    def test_result_has_required_keys(self):
        diffs = RNG.normal(0, 1, 20)
        result = TestSelector.select_paired(diffs)
        required = {"selected_test", "rationale", "normality_p", "normality_passed"}
        assert required.issubset(set(result.keys()))

    def test_extreme_outliers_trigger_wilcoxon(self):
        """Confirm outlier detection actually triggers Wilcoxon when outliers present."""
        diffs = np.array([0.1, 0.05, -0.05, 0.08, -0.02, 200.0])  # one extreme outlier
        result = TestSelector.select_paired(diffs)
        assert result["extreme_outliers_count"] >= 1


class TestSelectGroups:
    def test_equal_variance_selects_independent_t(self):
        a = RNG.normal(0, 1, 50)
        b = RNG.normal(0, 1, 50)
        result = TestSelector.select_groups([a, b])
        assert result["selected_test"] in {"independent_t", "welch_t"}
        assert "levene_p" in result

    def test_wrong_number_of_groups_raises(self):
        with pytest.raises(ValueError, match="exactly 2 groups"):
            TestSelector.select_groups([np.array([1.0, 2.0])])

    def test_result_has_required_keys(self):
        a = RNG.normal(0, 1, 30)
        b = RNG.normal(0, 2, 30)
        result = TestSelector.select_groups([a, b])
        required = {"selected_test", "rationale", "levene_p", "equal_variance_passed"}
        assert required.issubset(set(result.keys()))
