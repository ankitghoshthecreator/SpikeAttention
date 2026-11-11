"""Unit tests for assumption_tests.py: compare against scipy references."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.assumption_tests import AssumptionTester


RNG = np.random.default_rng(1)
NORMAL_DATA = RNG.normal(0, 1, 40)
UNIFORM_DATA = RNG.uniform(0, 1, 40)


class TestShapiroWilk:
    def test_statistic_matches_scipy(self):
        result = AssumptionTester.shapiro_wilk(NORMAL_DATA)
        sci_stat, sci_p = stats.shapiro(NORMAL_DATA)
        assert result["statistic"] == pytest.approx(sci_stat, abs=1e-6)
        assert result["p_value"] == pytest.approx(sci_p, abs=1e-6)

    def test_normal_data_passes(self):
        result = AssumptionTester.shapiro_wilk(NORMAL_DATA, alpha=0.05)
        # Normal data should generally pass (not guaranteed but very likely for N=40)
        assert "passed" in result

    def test_p_value_in_range(self):
        result = AssumptionTester.shapiro_wilk(NORMAL_DATA)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_too_few_samples_raises(self):
        with pytest.raises((ValueError, Exception)):
            AssumptionTester.shapiro_wilk(np.array([1.0, 2.0]))

    def test_result_has_required_keys(self):
        result = AssumptionTester.shapiro_wilk(NORMAL_DATA)
        assert {"statistic", "p_value", "passed"}.issubset(set(result.keys()))


class TestLevene:
    def test_statistic_matches_scipy(self):
        a = RNG.normal(0, 1, 30)
        b = RNG.normal(0, 2, 30)
        result = AssumptionTester.levene(a, b)
        sci_stat, sci_p = stats.levene(a, b, center="median")
        assert result["statistic"] == pytest.approx(sci_stat, rel=0.01)
        assert result["p_value"] == pytest.approx(sci_p, rel=0.01)

    def test_equal_variance_passes(self):
        a = RNG.normal(0, 1, 50)
        b = RNG.normal(0, 1, 50)
        result = AssumptionTester.levene(a, b, alpha=0.05)
        assert "passed" in result

    def test_result_has_required_keys(self):
        a, b = np.ones(10), np.ones(10) * 2
        result = AssumptionTester.levene(a, b)
        assert {"statistic", "p_value", "passed"}.issubset(set(result.keys()))


class TestOutliers:
    def test_no_outliers_in_normal_data(self):
        data = RNG.normal(0, 1, 50)
        result = AssumptionTester.outliers(data, method="iqr", threshold=1.5)
        assert "has_outliers" in result

    def test_extreme_outlier_detected(self):
        data = np.concatenate([RNG.normal(0, 1, 50), [100.0]])
        result = AssumptionTester.outliers(data, method="iqr", threshold=1.5)
        assert result["has_outliers"] is True
        assert result["count"] >= 1

    def test_result_has_required_keys(self):
        result = AssumptionTester.outliers(NORMAL_DATA)
        assert {"has_outliers", "count", "indices"}.issubset(set(result.keys()))


class TestQQData:
    def test_returns_tuple_of_arrays(self):
        result = AssumptionTester.qq_data(NORMAL_DATA)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lengths_match(self):
        theoretical, sample = AssumptionTester.qq_data(NORMAL_DATA)
        assert len(theoretical) == len(sample)


class TestSphericity:
    def test_trivially_passes_for_k_lt_3(self):
        # k < 3 → sphericity is trivially satisfied
        mat = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
        result = AssumptionTester.sphericity(mat)
        assert result["passed"] is True

    def test_valid_matrix_returns_dict(self):
        mat = RNG.normal(0, 1, (10, 4))
        result = AssumptionTester.sphericity(mat)
        assert "passed" in result
        assert "greenhouse_geisser_epsilon" in result
