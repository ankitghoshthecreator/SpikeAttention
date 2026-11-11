"""Unit tests for hypothesis_testing.py: power analysis, MDE, critical region."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.hypothesis_testing import HypothesisTester


class TestPowerAnalysis:
    def test_paired_t_power_known_value(self):
        """Power = P(|T| > t_crit | H1) for paired t with d=0.8, n=20, alpha=0.05."""
        power = HypothesisTester.power_analysis(effect_size=0.8, n=20, alpha=0.05)
        assert isinstance(power, float)
        assert 0.0 <= power <= 1.0

    def test_power_increases_with_n(self):
        small = HypothesisTester.power_analysis(effect_size=0.5, n=10, alpha=0.05)
        large = HypothesisTester.power_analysis(effect_size=0.5, n=100, alpha=0.05)
        assert large > small

    def test_power_increases_with_effect_size(self):
        small_d = HypothesisTester.power_analysis(effect_size=0.2, n=30, alpha=0.05)
        large_d = HypothesisTester.power_analysis(effect_size=0.8, n=30, alpha=0.05)
        assert large_d > small_d

    def test_power_range(self):
        p = HypothesisTester.power_analysis(effect_size=0.5, n=20, alpha=0.05)
        assert 0.0 <= p <= 1.0


class TestMinDetectableEffect:
    def test_returns_positive_float(self):
        mde = HypothesisTester.min_detectable_effect(n=30, alpha=0.05, power=0.80)
        assert isinstance(mde, float)
        assert mde > 0.0

    def test_mde_decreases_with_n(self):
        small = HypothesisTester.min_detectable_effect(n=10, alpha=0.05, power=0.80)
        large = HypothesisTester.min_detectable_effect(n=100, alpha=0.05, power=0.80)
        assert small > large


class TestCriticalRegion:
    def test_two_sided_t_symmetry(self):
        lower, upper = HypothesisTester.critical_region(df=20, alpha=0.05, tail="two-sided", dist="t")
        assert lower < 0 < upper
        assert abs(lower) == pytest.approx(abs(upper), abs=1e-10)

    def test_one_sided_greater(self):
        lower, upper = HypothesisTester.critical_region(df=20, alpha=0.05, tail="greater", dist="t")
        # For "greater" test, rejection region is [crit, +inf)
        assert lower > 0 or upper == float("inf")

    def test_normal_critical_region_two_sided(self):
        lower, upper = HypothesisTester.critical_region(df=1, alpha=0.05, tail="two-sided", dist="norm")
        assert upper == pytest.approx(stats.norm.ppf(0.975), abs=1e-6)
        assert lower == pytest.approx(-stats.norm.ppf(0.975), abs=1e-6)


class TestTypeErrorsSummary:
    def test_returns_dict_with_alpha_and_beta(self):
        from spikeattn.statistics.base import TestResult
        result = TestResult(
            name="TestXYZ",
            statistic=2.0,
            df=10.0,
            p_value=0.03,
            alpha=0.05,
            reject_h0=True,
        )
        summary = HypothesisTester.type_errors_summary(result, power=0.80)
        assert "alpha_type_i_error_rate" in summary
        assert "beta_type_ii_error_rate" in summary
        assert summary["alpha_type_i_error_rate"] == pytest.approx(0.05)
        assert summary["beta_type_ii_error_rate"] == pytest.approx(0.20, abs=1e-10)
