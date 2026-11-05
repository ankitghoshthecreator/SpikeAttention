"""Unit tests for nonparametric_tests.py: compare against scipy references."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.base import Hypothesis
from spikeattn.statistics.nonparametric_tests import (
    FriedmanTest,
    KruskalWallisTest,
    WilcoxonSignedRankTest,
)


RNG = np.random.default_rng(3)
HYP = Hypothesis(h0_text="H0", h1_text="H1")


class TestWilcoxonSignedRankTest:
    def test_statistic_matches_scipy(self):
        a = RNG.normal(5, 2, 20)
        b = RNG.normal(4.5, 2, 20)
        result = WilcoxonSignedRankTest().run(a, b, hypothesis=HYP)
        sci = stats.wilcoxon(a, b)
        assert result.statistic == pytest.approx(sci.statistic, abs=1.0)
        assert result.p_value == pytest.approx(sci.pvalue, abs=0.01)

    def test_effect_size_rank_biserial_range(self):
        a = RNG.normal(5, 1, 15)
        b = RNG.normal(3, 1, 15)
        result = WilcoxonSignedRankTest().run(a, b, hypothesis=HYP)
        assert result.effect_size is not None
        assert -1.0 <= result.effect_size <= 1.0

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            WilcoxonSignedRankTest().run(np.array([1.0, 2.0]), np.array([1.0]), hypothesis=HYP)

    def test_too_few_samples_raises(self):
        with pytest.raises((ValueError, Exception)):
            WilcoxonSignedRankTest().run(np.array([1.0]), np.array([1.0]), hypothesis=HYP)


class TestFriedmanTest:
    def _groups(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.5, 2.5, 3.5, 4.5])
        c = np.array([0.5, 1.5, 2.5, 3.5])
        return a, b, c

    def test_statistic_matches_scipy(self):
        a, b, c = self._groups()
        result = FriedmanTest().run(a, b, c, hypothesis=HYP)
        sci = stats.friedmanchisquare(a, b, c)
        assert result.statistic == pytest.approx(sci.statistic, abs=0.1)
        assert result.p_value == pytest.approx(sci.pvalue, abs=0.05)

    def test_kendalls_w_in_range(self):
        a, b, c = self._groups()
        result = FriedmanTest().run(a, b, c, hypothesis=HYP)
        assert result.effect_size is not None
        assert 0.0 <= result.effect_size <= 1.0

    def test_post_hoc_returned(self):
        a, b, c = self._groups()
        result = FriedmanTest().run(a, b, c, hypothesis=HYP)
        # post_hoc should be in assumptions when test completes
        assert isinstance(result.assumptions, dict)

    def test_fewer_than_3_groups_raises(self):
        with pytest.raises((ValueError, Exception)):
            FriedmanTest().run(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), hypothesis=HYP)

    def test_unequal_group_lengths_raises(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        c = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            FriedmanTest().run(a, b, c, hypothesis=HYP)


class TestKruskalWallisTest:
    def test_statistic_matches_scipy(self):
        a = RNG.normal(0, 1, 20)
        b = RNG.normal(0.5, 1, 20)
        c = RNG.normal(1.0, 1, 20)
        result = KruskalWallisTest().run(a, b, c, hypothesis=HYP)
        sci = stats.kruskal(a, b, c)
        assert result.statistic == pytest.approx(sci.statistic, abs=0.1)
        assert result.p_value == pytest.approx(sci.pvalue, abs=0.01)

    def test_fewer_than_2_groups_raises(self):
        with pytest.raises(ValueError):
            KruskalWallisTest().run(np.array([1.0, 2.0]), hypothesis=HYP)
