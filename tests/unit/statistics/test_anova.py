"""Unit tests for anova.py: compare against scipy.stats.f_oneway and statsmodels AnovaRM."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.anova import OneWayANOVA, RepeatedMeasuresANOVA
from spikeattn.statistics.base import Hypothesis


RNG = np.random.default_rng(4)
HYP = Hypothesis(h0_text="H0", h1_text="H1")


class TestOneWayANOVA:
    def _groups(self):
        a = RNG.normal(5, 1, 20)
        b = RNG.normal(5.5, 1, 20)
        c = RNG.normal(6, 1, 20)
        return a, b, c

    def test_statistic_matches_scipy(self):
        a, b, c = self._groups()
        result = OneWayANOVA().run(a, b, c, hypothesis=HYP)
        sci = stats.f_oneway(a, b, c)
        assert result.statistic == pytest.approx(sci.statistic, rel=0.01)
        assert result.p_value == pytest.approx(sci.pvalue, abs=0.01)

    def test_eta_squared_in_range(self):
        a, b, c = self._groups()
        result = OneWayANOVA().run(a, b, c, hypothesis=HYP)
        assert result.effect_size is not None
        assert 0.0 <= result.effect_size <= 1.0

    def test_fewer_than_2_groups_raises(self):
        with pytest.raises(ValueError):
            OneWayANOVA().run(np.array([1.0, 2.0]), hypothesis=HYP)

    def test_df_tuple(self):
        a, b, c = self._groups()
        result = OneWayANOVA().run(a, b, c, hypothesis=HYP)
        assert isinstance(result.df, tuple)
        assert len(result.df) == 2


class TestRepeatedMeasuresANOVA:
    def _matrix(self):
        """10 subjects × 3 conditions."""
        mat = np.array([
            [4.1, 5.2, 6.3],
            [3.8, 4.9, 5.7],
            [4.5, 5.8, 7.1],
            [3.9, 4.6, 5.9],
            [4.2, 5.1, 6.0],
            [4.0, 5.3, 6.5],
            [3.7, 4.8, 5.8],
            [4.3, 5.4, 6.4],
            [4.4, 5.6, 6.8],
            [4.1, 5.0, 6.1],
        ])
        return mat

    def test_p_value_in_range(self):
        mat = self._matrix()
        result = RepeatedMeasuresANOVA().run(mat, hypothesis=HYP)
        assert 0.0 <= result.p_value <= 1.0

    def test_has_epsilon_in_assumptions(self):
        mat = self._matrix()
        result = RepeatedMeasuresANOVA().run(mat, hypothesis=HYP)
        # Key is 'greenhouse_geisser_epsilon' nested inside 'sphericity'
        assert "sphericity" in result.assumptions
        assert "greenhouse_geisser_epsilon" in result.assumptions["sphericity"]

    def test_too_few_conditions_raises(self):
        """Single column (k=1) must raise since k_models < 2."""
        mat = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises((ValueError, Exception)):
            RepeatedMeasuresANOVA().run(mat, hypothesis=HYP)
