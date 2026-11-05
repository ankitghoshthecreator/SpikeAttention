"""Unit tests for regression.py: OLS simple and multiple fit, diagnostics."""

import numpy as np
import pytest

from spikeattn.statistics.regression import RegressionAnalyzer


RNG = np.random.default_rng(9)
N = 50
X1 = RNG.uniform(0, 10, N)
Y = 2.5 * X1 + 3.0 + RNG.normal(0, 1, N)


class TestFitSimple:
    def test_slope_close_to_true(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        # True slope ≈ 2.5, intercept ≈ 3.0
        assert result["slope"] == pytest.approx(2.5, abs=0.5)
        assert result["intercept"] == pytest.approx(3.0, abs=1.0)

    def test_result_has_required_keys(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        required = {"slope", "intercept", "r_squared", "n"}
        assert required.issubset(set(result.keys()))

    def test_r_squared_in_range(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_n_matches_input(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        assert result["n"] == N

    def test_unequal_lengths_raises(self):
        with pytest.raises((ValueError, Exception)):
            RegressionAnalyzer.fit_simple(X1[:10], Y)


class TestFitMultiple:
    def test_runs_without_error(self):
        X2 = RNG.uniform(0, 5, N)
        result = RegressionAnalyzer.fit_multiple(
            np.column_stack([X1, X2]),
            Y,
            feature_names=["x1", "x2"],
        )
        assert "coefficients" in result

    def test_coefficients_has_expected_keys(self):
        X2 = RNG.uniform(0, 5, N)
        result = RegressionAnalyzer.fit_multiple(
            np.column_stack([X1, X2]),
            Y,
            feature_names=["x1", "x2"],
        )
        assert "const" in result["coefficients"]
        assert "x1" in result["coefficients"]

    def test_r_squared_in_range(self):
        X2 = RNG.uniform(0, 5, N)
        result = RegressionAnalyzer.fit_multiple(
            np.column_stack([X1, X2]),
            Y,
            feature_names=["x1", "x2"],
        )
        assert 0.0 <= result["r_squared"] <= 1.0


class TestDiagnostics:
    def test_returns_dict(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        diag = RegressionAnalyzer.diagnostics(result)
        assert isinstance(diag, dict)

    def test_has_homoscedasticity_key(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        diag = RegressionAnalyzer.diagnostics(result)
        assert "homoscedasticity" in diag

    def test_has_normality_key(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        diag = RegressionAnalyzer.diagnostics(result)
        assert "normality" in diag

    def test_has_vif_key(self):
        result = RegressionAnalyzer.fit_simple(X1, Y)
        diag = RegressionAnalyzer.diagnostics(result)
        assert "vif" in diag
