"""Unit tests for effect_size.py: verify against hand calculations and scipy."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.effect_size import EffectSizeAnalyzer
from spikeattn.exceptions import InsufficientDataError


RNG = np.random.default_rng(6)


class TestCohensD:
    def test_known_value(self):
        """d = (mean_a - mean_b) / pooled_sd — hand verified."""
        a = np.array([5.0, 6.0, 7.0, 8.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        d = EffectSizeAnalyzer.cohens_d(a, b)
        # pooled SD = sqrt(((3*var_a + 3*var_b) / 6)) = sqrt(var) for equal groups
        s1 = np.std(a, ddof=1)
        s2 = np.std(b, ddof=1)
        n = len(a)
        sp = np.sqrt(((n - 1) * s1 ** 2 + (n - 1) * s2 ** 2) / (n + n - 2))
        expected = (np.mean(a) - np.mean(b)) / sp
        assert d == pytest.approx(expected, abs=1e-10)

    def test_zero_difference_gives_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert EffectSizeAnalyzer.cohens_d(a, a) == pytest.approx(0.0)

    def test_too_few_samples_raises(self):
        with pytest.raises(InsufficientDataError):
            EffectSizeAnalyzer.cohens_d([1.0], [2.0])


class TestPairedD:
    def test_known_value(self):
        diffs = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
        d_z = EffectSizeAnalyzer.paired_d(diffs)
        expected = np.mean(diffs) / np.std(diffs, ddof=1)
        assert d_z == pytest.approx(expected, abs=1e-10)


class TestHedgesG:
    def test_correction_factor(self):
        diffs = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
        n = len(diffs)
        df = n - 1
        d_z = EffectSizeAnalyzer.paired_d(diffs)
        correction = 1.0 - (3.0 / (4.0 * df - 1.0))
        expected_g = d_z * correction
        g = EffectSizeAnalyzer.hedges_g(diffs)
        assert g == pytest.approx(expected_g, abs=1e-10)

    def test_g_smaller_than_d_for_small_n(self):
        """Hedges' g < Cohen's d_z due to small-sample correction."""
        diffs = np.array([1.0, 1.5, 0.8, 1.2])
        d_z = EffectSizeAnalyzer.paired_d(diffs)
        g = EffectSizeAnalyzer.hedges_g(diffs)
        # For n=4, correction < 1.0, so |g| < |d_z|
        assert abs(g) < abs(d_z)


class TestRankBiserial:
    def test_manual_verification(self):
        """r_rb = (W+ - W-) / (W+ + W-)."""
        diffs = np.array([1.0, -0.5, 2.0, -1.0, 3.0])
        r_rb = EffectSizeAnalyzer.rank_biserial(diffs)
        assert -1.0 <= r_rb <= 1.0

    def test_all_positive_gives_positive(self):
        diffs = np.array([0.1, 0.5, 1.0, 2.0, 0.3])
        r_rb = EffectSizeAnalyzer.rank_biserial(diffs)
        assert r_rb > 0

    def test_all_zero_diffs_gives_zero(self):
        diffs = np.zeros(5)
        assert EffectSizeAnalyzer.rank_biserial(diffs) == pytest.approx(0.0)


class TestKendallsW:
    def test_perfect_agreement(self):
        """All raters rank identically → W=1."""
        mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)
        w = EffectSizeAnalyzer.kendalls_w(mat)
        assert w == pytest.approx(1.0, abs=1e-6)

    def test_range_01(self):
        mat = RNG.normal(0, 1, (5, 4))
        w = EffectSizeAnalyzer.kendalls_w(mat)
        assert 0.0 <= w <= 1.0


class TestCramersV:
    def test_matches_manual(self):
        table = np.array([[40, 20], [10, 30]])
        v = EffectSizeAnalyzer.cramers_v(table)
        from scipy.stats.contingency import chi2_contingency
        chi2 = chi2_contingency(table, correction=False).statistic
        n = np.sum(table)
        min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
        expected = np.sqrt(chi2 / (n * min_dim))
        assert v == pytest.approx(expected, abs=1e-10)


class TestEtaSquared:
    def test_zero_total_gives_zero(self):
        assert EffectSizeAnalyzer.eta_squared(0.0, 0.0) == pytest.approx(0.0)

    def test_full_between_gives_one(self):
        assert EffectSizeAnalyzer.eta_squared(10.0, 10.0) == pytest.approx(1.0)

    def test_in_range(self):
        val = EffectSizeAnalyzer.eta_squared(5.0, 20.0)
        assert 0.0 <= val <= 1.0


class TestInterpret:
    def test_negligible_d(self):
        label = EffectSizeAnalyzer.interpret(0.1, metric="d")
        assert "negligible" in label

    def test_large_d(self):
        label = EffectSizeAnalyzer.interpret(0.9, metric="d")
        assert "large" in label

    def test_sesoi_exceeds(self):
        label = EffectSizeAnalyzer.interpret(0.5, metric="d", sesoi=0.3)
        assert "practically meaningful" in label

    def test_sesoi_below(self):
        label = EffectSizeAnalyzer.interpret(0.1, metric="d", sesoi=0.3)
        assert "not practically meaningful" in label
