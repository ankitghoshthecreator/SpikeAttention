"""Effect size estimation, small-sample corrections, confidence intervals, and SESOI comparisons."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from spikeattn.exceptions import InsufficientDataError


class EffectSizeAnalyzer:
    """Computes standardized effect sizes with bias corrections and SESOI benchmarks."""

    @staticmethod
    def cohens_d(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
        """Compute Cohen's d for independent samples with pooled standard deviation."""
        a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            raise InsufficientDataError("Cohen's d requires at least 2 samples per group")
        s1_sq, s2_sq = np.var(a, ddof=1), np.var(b, ddof=1)
        sp = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
        if sp <= 1e-12:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / sp)

    @staticmethod
    def paired_d(d: np.ndarray | list[float]) -> float:
        """Compute paired Cohen's d_z = mean(diff) / sd(diff)."""
        diffs = np.asarray(d, dtype=np.float64)
        if len(diffs) < 2:
            raise InsufficientDataError("Paired d_z requires at least 2 pairs")
        s = np.std(diffs, ddof=1)
        if s <= 1e-12:
            return 0.0
        return float(np.mean(diffs) / s)

    @staticmethod
    def hedges_g(d: np.ndarray | list[float]) -> float:
        """Compute Hedges' g for paired differences with exact small-sample correction."""
        diffs = np.asarray(d, dtype=np.float64)
        n = len(diffs)
        if n < 2:
            raise InsufficientDataError("Hedges' g requires at least 2 pairs")
        d_z = EffectSizeAnalyzer.paired_d(diffs)
        df = n - 1
        correction = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
        return float(d_z * correction)

    @staticmethod
    def rank_biserial(diffs: np.ndarray | list[float]) -> float:
        """Compute the rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-) for Wilcoxon."""
        d = np.asarray(diffs, dtype=np.float64)
        non_zero = d[np.abs(d) > 1e-12]
        if len(non_zero) == 0:
            return 0.0
        ranks = stats.rankdata(np.abs(non_zero))
        pos_ranks = np.sum(ranks[non_zero > 0])
        neg_ranks = np.sum(ranks[non_zero < 0])
        total = pos_ranks + neg_ranks
        return float((pos_ranks - neg_ranks) / total) if total > 0 else 0.0

    @staticmethod
    def kendalls_w(matrix: np.ndarray) -> float:
        """Compute Kendall's W coefficient of concordance for Friedman test: W = stat / (N * (k - 1))."""
        mat = np.asarray(matrix, dtype=np.float64)
        n, k = mat.shape
        if n < 2 or k < 2:
            return 0.0
        ranked = np.apply_along_axis(stats.rankdata, 1, mat)
        r_sums = np.sum(ranked, axis=0)
        mean_r = np.mean(r_sums)
        s = np.sum((r_sums - mean_r) ** 2)
        w = float((12.0 * s) / ((n ** 2) * (k ** 3 - k)))
        return max(0.0, min(1.0, w))

    @staticmethod
    def cramers_v(table: np.ndarray) -> float:
        """Compute Cramér's V measure of association for contingency tables."""
        tab = np.asarray(table, dtype=np.float64)
        chi2_res = stats.contingency.chi2_contingency(tab, correction=False)
        n = np.sum(tab)
        r, c = tab.shape
        min_dim = min(r - 1, c - 1)
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2_res.statistic / (n * min_dim)))

    @staticmethod
    def eta_squared(ss_between: float, ss_total: float) -> float:
        """Compute Eta-squared (eta^2) effect size for ANOVA."""
        if ss_total <= 0:
            return 0.0
        return float(max(0.0, min(1.0, ss_between / ss_total)))

    @staticmethod
    def with_ci(
        effect_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        level: float = 0.95,
        B: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Compute an effect size with a bootstrap confidence interval."""
        arr = np.asarray(data)
        point_estimate = float(effect_fn(arr))
        rng = np.random.default_rng(seed)
        n = len(arr)
        boot_estimates = np.empty(B, dtype=np.float64)
        for i in range(B):
            idx = rng.integers(0, n, size=n)
            boot_estimates[i] = effect_fn(arr[idx])

        alpha = 1.0 - level
        lower = float(np.percentile(boot_estimates, 100.0 * (alpha / 2.0)))
        upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha / 2.0)))
        return point_estimate, lower, upper

    @staticmethod
    def interpret(magnitude: float, metric: str = "d", sesoi: float | None = None) -> str:
        """Classify effect magnitude into conventional benchmarks alongside SESOI comparison."""
        val = abs(magnitude)
        if metric in {"d", "g", "d_z"}:
            if val < 0.20:
                mag_label = "negligible"
            elif val < 0.50:
                mag_label = "small"
            elif val < 0.80:
                mag_label = "medium"
            else:
                mag_label = "large"
        elif metric in {"r", "rank_biserial"}:
            if val < 0.10:
                mag_label = "negligible"
            elif val < 0.30:
                mag_label = "small"
            elif val < 0.50:
                mag_label = "medium"
            else:
                mag_label = "large"
        else:
            mag_label = f"value={magnitude:.4f}"

        sesoi_note = ""
        if sesoi is not None:
            if val >= sesoi:
                sesoi_note = f" (exceeds SESOI={sesoi:.3f}; practically meaningful)"
            else:
                sesoi_note = f" (below SESOI={sesoi:.3f}; not practically meaningful)"

        return f"{mag_label}{sesoi_note}"
