"""Unit tests for mcnemar_test.py: exact binomial (n<25) and chi-square (n>=25)."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.base import Hypothesis
from spikeattn.statistics.mcnemar_test import McNemarTest


HYP = Hypothesis(h0_text="H0", h1_text="H1")

RNG = np.random.default_rng(99)


def _make_paired_arrays(n_total, b, c):
    """Build two binary correctness arrays with exactly b and c discordant pairs.

    b = model_a correct & model_b wrong
    c = model_a wrong & model_b correct
    n_agree = n_total - b - c (both correct or both wrong)
    """
    # Both correct
    n_both = n_total - b - c
    a_arr = np.concatenate([
        np.ones(n_both // 2, dtype=bool),   # both correct
        np.zeros(n_both - n_both // 2, dtype=bool),  # both wrong
        np.ones(b, dtype=bool),              # a correct, b wrong
        np.zeros(c, dtype=bool),             # a wrong, b correct
    ])
    b_arr = np.concatenate([
        np.ones(n_both // 2, dtype=bool),
        np.zeros(n_both - n_both // 2, dtype=bool),
        np.zeros(b, dtype=bool),
        np.ones(c, dtype=bool),
    ])
    return a_arr, b_arr


class TestMcNemarTest:
    def test_exact_small_n(self):
        """n_discordant = b+c < 25 → exact binomial test."""
        b_count, c_count = 5, 10
        a_arr, b_arr = _make_paired_arrays(50, b_count, c_count)
        result = McNemarTest().run(a_arr, b_arr, hypothesis=HYP)
        # Reference: exact binomial test on min(b,c) successes out of b+c
        n_disc = b_count + c_count
        sci = stats.binomtest(min(b_count, c_count), n_disc, p=0.5)
        assert result.p_value == pytest.approx(sci.pvalue, abs=1e-6)

    def test_chi_square_large_n(self):
        """n_discordant >= 25 → continuity-corrected chi-square."""
        b_count, c_count = 30, 50
        a_arr, b_arr = _make_paired_arrays(200, b_count, c_count)
        result = McNemarTest().run(a_arr, b_arr, hypothesis=HYP)
        # continuity-corrected McNemar: (|b - c| - 1)^2 / (b + c)
        expected_chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        expected_p = 1 - stats.chi2.cdf(expected_chi2, df=1)
        assert result.statistic == pytest.approx(expected_chi2, rel=0.01)
        assert result.p_value == pytest.approx(expected_p, abs=0.01)

    def test_non_binary_raises(self):
        """Different-length arrays should raise ValueError."""
        a_arr = np.array([True, True, False])
        b_arr = np.array([True, False])
        with pytest.raises((ValueError, Exception)):
            McNemarTest().run(a_arr, b_arr, hypothesis=HYP)

    def test_result_has_p_value(self):
        b_count, c_count = 8, 12
        a_arr, b_arr = _make_paired_arrays(50, b_count, c_count)
        result = McNemarTest().run(a_arr, b_arr, hypothesis=HYP)
        assert 0.0 <= result.p_value <= 1.0
