"""Unit tests for multiple_comparison.py: compare against statsmodels multipletests."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from spikeattn.statistics.multiple_comparison import MultipleComparisonCorrector


P_VALS = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.396, 0.950]


class TestBonferroni:
    def test_matches_statsmodels(self):
        p = np.array(P_VALS)
        our = MultipleComparisonCorrector.bonferroni(p)
        _, ref, _, _ = multipletests(p, method="bonferroni")
        np.testing.assert_allclose(our, ref, atol=1e-10)

    def test_clamps_to_one(self):
        p = [0.5, 0.6, 0.7]
        adj = MultipleComparisonCorrector.bonferroni(p)
        assert all(v <= 1.0 for v in adj)

    def test_empty_input(self):
        result = MultipleComparisonCorrector.bonferroni([])
        assert len(result) == 0


class TestHolm:
    def test_matches_statsmodels(self):
        p = np.array(P_VALS)
        our = MultipleComparisonCorrector.holm(p)
        _, ref, _, _ = multipletests(p, method="holm")
        np.testing.assert_allclose(our, ref, atol=1e-8)

    def test_monotone_property(self):
        """Holm-adjusted values must be non-decreasing in sorted order."""
        p = np.array([0.01, 0.04, 0.09])
        adj = MultipleComparisonCorrector.holm(p)
        sorted_adj = np.sort(adj)
        assert all(sorted_adj[i] <= sorted_adj[i + 1] for i in range(len(sorted_adj) - 1))

    def test_empty_input(self):
        result = MultipleComparisonCorrector.holm([])
        assert len(result) == 0

    def test_single_value(self):
        p = [0.03]
        adj = MultipleComparisonCorrector.holm(p)
        assert adj[0] == pytest.approx(0.03)


class TestBenjaminiHochberg:
    def test_matches_statsmodels(self):
        p = np.array(P_VALS)
        our = MultipleComparisonCorrector.benjamini_hochberg(p)
        _, ref, _, _ = multipletests(p, method="fdr_bh")
        np.testing.assert_allclose(our, ref, atol=1e-8)

    def test_clamps_to_one(self):
        p = [0.9, 0.95, 0.99]
        adj = MultipleComparisonCorrector.benjamini_hochberg(p)
        assert all(v <= 1.0 for v in adj)

    def test_empty_input(self):
        result = MultipleComparisonCorrector.benjamini_hochberg([])
        assert len(result) == 0


class TestCorrectDispatch:
    def test_holm_dispatch(self):
        p = np.array([0.01, 0.05, 0.1])
        a = MultipleComparisonCorrector.correct(p, method="holm")
        b = MultipleComparisonCorrector.holm(p)
        np.testing.assert_allclose(a, b)

    def test_bonferroni_dispatch(self):
        p = np.array([0.01, 0.05, 0.1])
        a = MultipleComparisonCorrector.correct(p, method="bonferroni")
        b = MultipleComparisonCorrector.bonferroni(p)
        np.testing.assert_allclose(a, b)

    def test_bh_dispatch(self):
        p = np.array([0.01, 0.05, 0.1])
        a = MultipleComparisonCorrector.correct(p, method="bh")
        b = MultipleComparisonCorrector.benjamini_hochberg(p)
        np.testing.assert_allclose(a, b)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown correction"):
            MultipleComparisonCorrector.correct([0.05], method="fwer_xyz")


class TestCorrectFamilies:
    def test_multiple_families(self):
        families = {
            "primary": [0.01, 0.04, 0.08],
            "secondary": [0.02, 0.09],
        }
        methods = {"primary": "holm", "secondary": "bh"}
        result = MultipleComparisonCorrector.correct_families(families, methods)
        assert "primary" in result and "secondary" in result
        assert len(result["primary"]) == 3
        assert len(result["secondary"]) == 2
