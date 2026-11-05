"""Unit tests for correlation.py: compare against scipy references."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.correlation import CorrelationAnalyzer


RNG = np.random.default_rng(8)
X = RNG.normal(0, 1, 40)
Y = X * 0.8 + RNG.normal(0, 0.5, 40)


class TestPearson:
    def test_statistic_matches_scipy(self):
        result = CorrelationAnalyzer.pearson(X, Y)
        sci_r, sci_p = stats.pearsonr(X, Y)
        assert result["coefficient"] == pytest.approx(sci_r, abs=1e-10)
        assert result["p_value"] == pytest.approx(sci_p, abs=1e-10)

    def test_ci_contains_r(self):
        result = CorrelationAnalyzer.pearson(X, Y, alpha=0.05)
        r = result["coefficient"]
        lo, hi = result["ci"]
        assert lo <= r <= hi

    def test_self_correlation_is_one(self):
        result = CorrelationAnalyzer.pearson(X, X)
        assert result["coefficient"] == pytest.approx(1.0, abs=1e-6)

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            CorrelationAnalyzer.pearson(X, X[:5])


class TestSpearman:
    def test_statistic_matches_scipy(self):
        result = CorrelationAnalyzer.spearman(X, Y)
        sci_r, sci_p = stats.spearmanr(X, Y)
        assert result["coefficient"] == pytest.approx(sci_r, abs=1e-10)
        assert result["p_value"] == pytest.approx(sci_p, abs=1e-10)

    def test_monotone_data_gives_positive_rho(self):
        x = np.arange(10, dtype=float)
        y = x ** 2
        result = CorrelationAnalyzer.spearman(x, y)
        assert result["coefficient"] > 0.9


class TestMatrix:
    def test_diagonal_is_one(self):
        data = np.column_stack([X, Y, RNG.normal(0, 1, 40)])
        import pandas as pd
        df = pd.DataFrame(data, columns=["a", "b", "c"])
        mat = CorrelationAnalyzer.matrix(df)
        np.testing.assert_allclose(np.diag(mat.to_numpy()), np.ones(3), atol=1e-10)

    def test_shape(self):
        import pandas as pd
        df = pd.DataFrame(np.column_stack([X, Y]), columns=["x", "y"])
        mat = CorrelationAnalyzer.matrix(df)
        assert mat.shape == (2, 2)
