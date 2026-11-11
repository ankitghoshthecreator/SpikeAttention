"""Unit tests for chi_square.py: compare against scipy.stats.contingency."""

import numpy as np
import pytest
from scipy.stats.contingency import chi2_contingency
from scipy.stats import chisquare

from spikeattn.statistics.chi_square import ChiSquareTest
from spikeattn.statistics.base import Hypothesis


HYP = Hypothesis(h0_text="H0", h1_text="H1")


class TestChiSquareIndependence:
    def _table(self):
        return np.array([[50, 30], [20, 100]])

    def test_statistic_matches_scipy(self):
        table = self._table()
        test = ChiSquareTest()
        result = test.independence(table)
        sci = chi2_contingency(table, correction=False)
        assert result.statistic == pytest.approx(sci.statistic, abs=1e-6)
        assert result.p_value == pytest.approx(sci.pvalue, abs=1e-6)

    def test_cramers_v_in_range(self):
        table = self._table()
        result = ChiSquareTest().independence(table)
        assert result.effect_size is not None
        assert 0.0 <= result.effect_size <= 1.0

    def test_expected_counts_in_assumptions(self):
        """Cochran rule info should be in assumptions dict."""
        table = self._table()
        result = ChiSquareTest().independence(table)
        assert isinstance(result.assumptions, dict)
        assert "cochran_rule_passed" in result.assumptions

    def test_non_2d_raises(self):
        """1D input must raise ValueError/InsufficientDataError for independence test."""
        with pytest.raises((ValueError, Exception)):
            ChiSquareTest().independence(np.array([1.0, 2.0, 3.0]))


class TestChiSquareGoodnessOfFit:
    def test_uniform_expected(self):
        observed = np.array([20, 25, 22, 23])
        result = ChiSquareTest().goodness_of_fit(observed)
        sci = chisquare(observed)
        assert result.statistic == pytest.approx(sci.statistic, abs=1e-6)
        assert result.p_value == pytest.approx(sci.pvalue, abs=1e-6)

    def test_custom_expected(self):
        observed = np.array([20.0, 30.0, 50.0])
        expected = np.array([0.2, 0.3, 0.5])
        # Pass expected as proportions × n
        exp_counts = expected * np.sum(observed)
        result = ChiSquareTest().goodness_of_fit(observed, exp=exp_counts)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_in_range(self):
        observed = np.array([10.0, 15.0, 12.0, 13.0])
        result = ChiSquareTest().goodness_of_fit(observed)
        assert 0.0 <= result.p_value <= 1.0
