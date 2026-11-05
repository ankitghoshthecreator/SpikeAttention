"""Hypothesis testing registry, power analysis, and critical region calculations."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import optimize, stats

from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


class HypothesisTester:
    """Coordinates hypothesis tests, critical regions, and power calculations."""

    def __init__(self):
        self._registry: dict[str, BaseStatisticalTest] = {}

    def register(self, name: str, test: BaseStatisticalTest) -> None:
        """Register a statistical test instance."""
        self._registry[name.lower()] = test

    def get_test(self, name: str) -> BaseStatisticalTest:
        """Retrieve a registered test by name."""
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Test '{name}' not found in registry; available: {list(self._registry.keys())}")
        return self._registry[key]

    @staticmethod
    def critical_region(
        df: float,
        alpha: float = 0.05,
        tail: str = "two-sided",
        dist: str = "t",
    ) -> tuple[float, float]:
        """Compute the rejection region boundary values."""
        if dist == "t":
            d = stats.t(df=df)
        elif dist == "norm":
            d = stats.norm()
        elif dist == "chi2":
            d = stats.chi2(df=df)
            return (0.0, float(d.ppf(1.0 - alpha)))
        elif dist == "f":
            # df expected as tuple (df1, df2) if passed
            pass
        else:
            d = stats.t(df=df)

        if tail == "two-sided":
            crit = float(d.ppf(1.0 - alpha / 2.0))
            return (-crit, crit)
        elif tail == "greater":
            return (float(d.ppf(1.0 - alpha)), float("inf"))
        elif tail == "less":
            return (float("-inf"), float(d.ppf(alpha)))
        else:
            raise ValueError(f"Unknown tail: {tail}")

    @staticmethod
    def power_analysis(
        effect_size: float,
        n: int,
        alpha: float = 0.05,
        tail: str = "two-sided",
    ) -> float:
        """Compute post-hoc or prospective power for a paired/one-sample t-test.

        Uses the non-central t-distribution with non-centrality parameter delta = effect_size * sqrt(n).
        """
        if n < 2:
            return 0.0
        df = n - 1
        delta = float(abs(effect_size) * np.sqrt(n))

        if tail == "two-sided":
            crit = stats.t.ppf(1.0 - alpha / 2.0, df)
            power = stats.nct.sf(crit, df, delta) + stats.nct.cdf(-crit, df, delta)
        elif tail == "greater":
            crit = stats.t.ppf(1.0 - alpha, df)
            power = stats.nct.sf(crit, df, delta)
        elif tail == "less":
            crit = stats.t.ppf(alpha, df)
            power = stats.nct.cdf(crit, df, delta)
        else:
            raise ValueError(f"Unknown tail: {tail}")

        return float(max(0.0, min(1.0, power)))

    @staticmethod
    def min_detectable_effect(
        n: int,
        alpha: float = 0.05,
        power: float = 0.80,
        tail: str = "two-sided",
    ) -> float:
        """Calculate the minimum detectable effect size (MDE) given sample size n and target power."""
        if n < 2:
            return float("inf")

        def objective(es: float) -> float:
            return HypothesisTester.power_analysis(es, n=n, alpha=alpha, tail=tail) - power

        try:
            sol = optimize.root_scalar(objective, bracket=[1e-4, 10.0], method="brentq")
            return float(sol.root)
        except Exception:
            # Fallback normal approximation: delta approx (z_{1-alpha/2} + z_{power}) / sqrt(n)
            z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
            z_beta = stats.norm.ppf(power)
            return float((z_alpha + z_beta) / np.sqrt(n))

    @staticmethod
    def type_errors_summary(result: TestResult, power: float | None = None) -> dict[str, Any]:
        """Summarize Type I error alpha, decision, power, and Type II error beta."""
        beta = 1.0 - power if power is not None else None
        return {
            "test_name": result.name,
            "alpha_type_i_error_rate": result.alpha,
            "p_value": result.p_value,
            "decision": "Reject H0 (risk of Type I error: alpha)" if result.reject_h0 else "Fail to reject H0 (risk of Type II error: beta)",
            "power": power,
            "beta_type_ii_error_rate": beta,
        }
