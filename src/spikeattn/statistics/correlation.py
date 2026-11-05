"""Bivariate and multivariate correlation analysis: Pearson, Spearman, and adjusted matrices."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spikeattn.exceptions import InsufficientDataError
from spikeattn.statistics.multiple_comparison import MultipleComparisonCorrector


class CorrelationAnalyzer:
    """Computes bivariate linear and monotonic associations with inference and CIs."""

    @staticmethod
    def pearson(x: np.ndarray | list[float], y: np.ndarray | list[float], alpha: float = 0.05) -> dict[str, Any]:
        """Compute Pearson correlation r, p-value, and Fisher z-transform CI."""
        a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        n = len(a)
        if n < 3:
            raise InsufficientDataError("Pearson correlation requires at least 3 pairs")

        r, p_val = stats.pearsonr(a, b)

        # Fisher z-transformation confidence interval
        r_clamped = np.clip(r, -0.999999, 0.999999)
        z = np.arctanh(r_clamped)
        se_z = 1.0 / np.sqrt(n - 3)
        crit = stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lower = float(np.tanh(z - crit * se_z))
        ci_upper = float(np.tanh(z + crit * se_z))

        return {
            "method": "Pearson",
            "coefficient": float(r),
            "p_value": float(p_val),
            "alpha": float(alpha),
            "ci": (ci_lower, ci_upper),
            "n": n,
        }

    @staticmethod
    def spearman(x: np.ndarray | list[float], y: np.ndarray | list[float], alpha: float = 0.05) -> dict[str, Any]:
        """Compute Spearman rank-order correlation rho and p-value."""
        a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        n = len(a)
        if n < 3:
            raise InsufficientDataError("Spearman correlation requires at least 3 pairs")

        res = stats.spearmanr(a, b)
        return {
            "method": "Spearman",
            "coefficient": float(res.statistic),
            "p_value": float(res.pvalue),
            "alpha": float(alpha),
            "n": n,
        }

    @staticmethod
    def matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        """Compute correlation matrix across numeric columns."""
        num_df = df.select_dtypes(include=[np.number])
        return num_df.corr(method=method)

    @staticmethod
    def significance_matrix(
        df: pd.DataFrame,
        method: str = "pearson",
        correction: str = "holm",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pairwise correlation coefficient matrix and adjusted p-value matrix."""
        num_df = df.select_dtypes(include=[np.number])
        cols = list(num_df.columns)
        m = len(cols)
        corr_mat = np.eye(m, dtype=np.float64)
        raw_p_mat = np.zeros((m, m), dtype=np.float64)

        pairs = []
        raw_p_list = []
        for i in range(m):
            for j in range(i + 1, m):
                col_i, col_j = cols[i], cols[j]
                if method == "pearson":
                    res = CorrelationAnalyzer.pearson(num_df[col_i], num_df[col_j])
                else:
                    res = CorrelationAnalyzer.spearman(num_df[col_i], num_df[col_j])
                corr_mat[i, j] = corr_mat[j, i] = res["coefficient"]
                raw_p_mat[i, j] = raw_p_mat[j, i] = res["p_value"]
                pairs.append((i, j))
                raw_p_list.append(res["p_value"])

        adj_p_mat = np.zeros((m, m), dtype=np.float64)
        if raw_p_list:
            adj_p_list = MultipleComparisonCorrector.correct(raw_p_list, method=correction)
            for (i, j), adj_p in zip(pairs, adj_p_list, strict=True):
                adj_p_mat[i, j] = adj_p_mat[j, i] = adj_p

        corr_df = pd.DataFrame(corr_mat, index=cols, columns=cols)
        adj_p_df = pd.DataFrame(adj_p_mat, index=cols, columns=cols)
        return corr_df, adj_p_df
