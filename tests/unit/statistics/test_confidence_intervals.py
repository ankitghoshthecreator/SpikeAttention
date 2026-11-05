"""Unit tests for confidence_intervals.py: compare against statsmodels Wilson CI."""

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from spikeattn.statistics.confidence_intervals import ConfidenceIntervalEstimator


class TestWilsonCI:
    def test_matches_statsmodels(self):
        n, k = 100, 65
        lo, hi = ConfidenceIntervalEstimator.wilson(k, n, level=0.95)
        ref_lo, ref_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert lo == pytest.approx(ref_lo, abs=1e-6)
        assert hi == pytest.approx(ref_hi, abs=1e-6)

    def test_ci_contains_proportion(self):
        n, k = 200, 120
        lo, hi = ConfidenceIntervalEstimator.wilson(k, n, level=0.95)
        assert lo <= k / n <= hi

    def test_bounds_in_range(self):
        lo, hi = ConfidenceIntervalEstimator.wilson(10, 50)
        assert 0.0 <= lo <= hi <= 1.0

    def test_extreme_proportion_zero(self):
        lo, hi = ConfidenceIntervalEstimator.wilson(0, 50)
        assert lo >= 0.0
        assert hi >= 0.0


class TestNormalCI:
    def test_symmetry(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        lo, hi = ConfidenceIntervalEstimator.normal(mean, se, level=0.95)
        mid = (lo + hi) / 2
        assert mid == pytest.approx(mean, abs=1e-10)

    def test_width_decreases_with_n(self):
        small = np.random.default_rng(10).normal(0, 1, 10)
        large = np.random.default_rng(10).normal(0, 1, 1000)
        se_s = float(np.std(small, ddof=1) / np.sqrt(len(small)))
        se_l = float(np.std(large, ddof=1) / np.sqrt(len(large)))
        lo_s, hi_s = ConfidenceIntervalEstimator.normal(float(np.mean(small)), se_s)
        lo_l, hi_l = ConfidenceIntervalEstimator.normal(float(np.mean(large)), se_l)
        assert (hi_s - lo_s) > (hi_l - lo_l)


class TestBootstrapCI:
    def test_contains_true_mean(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10.0, 1.0, 50)
        lo, hi = ConfidenceIntervalEstimator.bootstrap(x, stat_fn=np.mean, level=0.95, B=500, seed=42)
        assert lo <= float(np.mean(x)) <= hi

    def test_bounds_in_sensible_range(self):
        x = np.ones(30) * 5.0
        lo, hi = ConfidenceIntervalEstimator.bootstrap(x, stat_fn=np.mean, B=200, seed=0)
        assert lo == pytest.approx(5.0, abs=1e-3)
        assert hi == pytest.approx(5.0, abs=1e-3)

    def test_requires_min_samples(self):
        with pytest.raises((ValueError, Exception)):
            ConfidenceIntervalEstimator.bootstrap(np.array([1.0]), stat_fn=np.mean, B=100, seed=0)


class TestDifferenceCI:
    def test_returns_two_floats(self):
        rng = np.random.default_rng(5)
        a = rng.normal(5, 1, 20)
        b = rng.normal(4, 1, 20)
        lo, hi = ConfidenceIntervalEstimator.difference_ci(a, b, level=0.95, B=200, seed=0)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi
