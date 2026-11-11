"""Unit tests for ttest.py: compare against scipy references."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.base import Hypothesis
from spikeattn.statistics.ttest import (
    CorrectedResampledTTest,
    IndependentTTest,
    OneSampleTTest,
    PairedTTest,
    WelchTTest,
)


RNG = np.random.default_rng(2)
HYP = Hypothesis(h0_text="No effect", h1_text="Effect exists")


class TestOneSampleTTest:
    def test_statistic_matches_scipy(self):
        x = RNG.normal(5, 2, 30)
        result = OneSampleTTest().run(x, hypothesis=HYP, unit="patient_group")
        sci_t, sci_p = stats.ttest_1samp(x, popmean=0.0)
        assert result.statistic == pytest.approx(sci_t, abs=1e-6)
        assert result.p_value == pytest.approx(sci_p, abs=1e-6)

    def test_known_result_reject(self):
        """Mean=10, std=1, n=30 — should reject H0 (mean≠0)."""
        x = RNG.normal(10, 1, 30)
        hyp = Hypothesis(h0_text="H0", h1_text="H1", alpha=0.05)
        result = OneSampleTTest().run(x, hypothesis=hyp)
        assert result.reject_h0 is True

    def test_custom_popmean(self):
        x = RNG.normal(5, 1, 20)
        hyp = Hypothesis(h0_text="H0", h1_text="H1")
        result = OneSampleTTest(mu0=5.0).run(x, hypothesis=hyp)
        sci_t, sci_p = stats.ttest_1samp(x, popmean=5.0)
        assert result.statistic == pytest.approx(sci_t, abs=1e-5)


class TestIndependentTTest:
    def test_statistic_matches_scipy(self):
        a = RNG.normal(0, 1, 25)
        b = RNG.normal(0, 1, 25)
        result = IndependentTTest().run(a, b, hypothesis=HYP)
        sci_t, sci_p = stats.ttest_ind(a, b, equal_var=True)
        assert result.statistic == pytest.approx(sci_t, abs=1e-6)
        assert result.p_value == pytest.approx(sci_p, abs=1e-6)


class TestWelchTTest:
    def test_statistic_matches_scipy(self):
        a = RNG.normal(0, 1, 20)
        b = RNG.normal(0, 3, 30)
        result = WelchTTest().run(a, b, hypothesis=HYP)
        sci_t, sci_p = stats.ttest_ind(a, b, equal_var=False)
        assert result.statistic == pytest.approx(sci_t, abs=1e-6)
        assert result.p_value == pytest.approx(sci_p, abs=1e-6)


class TestPairedTTest:
    def test_statistic_matches_scipy(self):
        a = RNG.normal(5, 2, 20)
        b = a + RNG.normal(0.5, 0.5, 20)
        result = PairedTTest().run(a, b, hypothesis=HYP)
        sci_t, sci_p = stats.ttest_rel(a, b)
        assert result.statistic == pytest.approx(sci_t, abs=1e-6)
        assert result.p_value == pytest.approx(sci_p, abs=1e-6)

    def test_effect_size_present(self):
        a = RNG.normal(5, 1, 15)
        b = a + RNG.normal(1, 0.3, 15)
        result = PairedTTest().run(a, b, hypothesis=HYP)
        assert result.effect_size is not None
        assert isinstance(result.effect_size, float)

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            PairedTTest().run(np.array([1.0, 2.0]), np.array([1.0]), hypothesis=HYP)


class TestCorrectedResampledTTest:
    def test_formula_manual_verification(self):
        """Verify Nadeau-Bengio formula: t = mean(d) / sqrt((1/K + n_test/n_train) * var(d))."""
        K = 5
        n_train, n_test = 32, 8
        diffs = np.array([0.02, 0.03, 0.01, 0.04, 0.02], dtype=np.float64)
        result = CorrectedResampledTTest(k_folds=K, n_train=n_train, n_test=n_test).run(
            diffs,
            hypothesis=HYP,
        )
        mean_d = np.mean(diffs)
        var_d = np.var(diffs, ddof=1)
        expected_t = mean_d / np.sqrt((1 / K + n_test / n_train) * var_d)
        assert result.statistic == pytest.approx(expected_t, rel=1e-5)

    def test_df_equals_K_minus_1(self):
        K = 10
        diffs = RNG.normal(0, 0.1, K)
        result = CorrectedResampledTTest(k_folds=K, n_train=100, n_test=20).run(
            diffs, hypothesis=HYP
        )
        assert result.df == pytest.approx(K - 1)

    def test_p_value_in_range(self):
        K = 5
        diffs = RNG.normal(0.01, 0.05, K)
        result = CorrectedResampledTTest(k_folds=K, n_train=40, n_test=10).run(
            diffs, hypothesis=HYP
        )
        assert 0.0 <= result.p_value <= 1.0
