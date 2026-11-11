"""Confidence interval estimators: Wilson, Normal approximation, and patient-clustered bootstrap."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from spikeattn.exceptions import InsufficientDataError


class ConfidenceIntervalEstimator:
    """Computes parametric, binomial, and clustered non-parametric confidence intervals."""

    @staticmethod
    def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
        """Wilson score interval for a binomial proportion.

        Calculates:
        p_tilde = (k + z^2 / 2) / (n + z^2)
        half_width = (z / (n + z^2)) * sqrt(k*(n - k)/n + z^2 / 4)
        """
        if n <= 0:
            raise InsufficientDataError("Sample size n must be positive for Wilson interval")
        if not (0 <= k <= n):
            raise ValueError(f"k must be in [0, {n}], got {k}")
        if not (0.0 < level < 1.0):
            raise ValueError(f"Confidence level must be in (0, 1), got {level}")

        alpha = 1.0 - level
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z2 = z ** 2

        p_tilde = (k + z2 / 2.0) / (n + z2)
        margin = (z / (n + z2)) * np.sqrt((k * (n - k)) / n + z2 / 4.0)

        lower = max(0.0, float(p_tilde - margin))
        upper = min(1.0, float(p_tilde + margin))
        return lower, upper

    @staticmethod
    def normal(mean: float, se: float, level: float = 0.95) -> tuple[float, float]:
        """Normal approximation confidence interval: mean +/- z * se."""
        if se < 0:
            raise ValueError("Standard error cannot be negative")
        alpha = 1.0 - level
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return float(mean - z * se), float(mean + z * se)

    @staticmethod
    def bootstrap(
        data: np.ndarray,
        stat_fn: Callable[[np.ndarray], float],
        cluster_ids: np.ndarray | None = None,
        B: int = 2000,
        level: float = 0.95,
        method: str = "percentile",
        seed: int = 42,
    ) -> tuple[float, float]:
        """Patient-clustered or standard bootstrap confidence interval.

        When cluster_ids is provided, bootstrap resamples clusters (patients) with replacement,
        preserving all within-cluster dependencies.
        """
        arr = np.asarray(data)
        n = len(arr)
        if n < 2:
            raise InsufficientDataError("Bootstrap requires at least 2 observations")
        if not (0.0 < level < 1.0):
            raise ValueError("Confidence level must be in (0, 1)")

        rng = np.random.default_rng(seed)
        alpha = 1.0 - level
        boot_stats = np.empty(B, dtype=np.float64)

        if cluster_ids is not None:
            clusters = np.asarray(cluster_ids)
            if len(clusters) != n:
                raise ValueError("cluster_ids must match data length")
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters)

            # Pre-index data by cluster for speed
            cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

            for b in range(B):
                sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
                sample_idx = np.concatenate([cluster_indices[c] for c in sampled_clusters])
                boot_stats[b] = stat_fn(arr[sample_idx])
        else:
            for b in range(B):
                sample_idx = rng.integers(0, n, size=n)
                boot_stats[b] = stat_fn(arr[sample_idx])

        boot_stats = boot_stats[np.isfinite(boot_stats)]
        if len(boot_stats) < B // 2:
            raise InsufficientDataError("More than half of bootstrap replicates produced non-finite results")

        if method == "percentile":
            lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
            upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
        elif method == "bca":
            orig = stat_fn(arr)
            z0 = float(stats.norm.ppf(np.mean(boot_stats < orig)))
            # Standard percentile fallback if extreme z0
            if not np.isfinite(z0):
                lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
                upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
            else:
                z_alpha = float(stats.norm.ppf(alpha / 2.0))
                z_1_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
                a1 = float(stats.norm.cdf(z0 + (z0 + z_alpha)))
                a2 = float(stats.norm.cdf(z0 + (z0 + z_1_alpha)))
                lower = float(np.percentile(boot_stats, 100.0 * a1))
                upper = float(np.percentile(boot_stats, 100.0 * a2))
        else:
            raise ValueError(f"Unknown bootstrap method: '{method}'")

        return lower, upper

    @staticmethod
    def difference_ci(
        a: np.ndarray,
        b: np.ndarray,
        paired: bool = True,
        cluster_ids: np.ndarray | None = None,
        level: float = 0.95,
        B: int = 2000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Confidence interval for the difference between two estimators."""
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)

        if paired:
            if len(arr_a) != len(arr_b):
                raise ValueError("Paired arrays must have identical length")
            diffs = arr_a - arr_b
            return ConfidenceIntervalEstimator.bootstrap(
                diffs, stat_fn=np.mean, cluster_ids=cluster_ids, B=B, level=level, seed=seed
            )
        else:
            rng = np.random.default_rng(seed)
            boot_diffs = np.empty(B, dtype=np.float64)
            na, nb = len(arr_a), len(arr_b)
            for i in range(B):
                sa = rng.choice(arr_a, size=na, replace=True)
                sb = rng.choice(arr_b, size=nb, replace=True)
                boot_diffs[i] = np.mean(sa) - np.mean(sb)
            alpha = 1.0 - level
            lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
            upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))
            return lower, upper
