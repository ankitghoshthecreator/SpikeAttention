"""Parametric Student and Welch t-tests, paired t-test, and corrected resampled t-test."""

from typing import Any

import numpy as np
from scipy import stats

from spikeattn.exceptions import InsufficientDataError
from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


def _p_value_from_t(stat: float, df: float, tail: str) -> float:
    """Compute tail-appropriate p-value from t-statistic and degrees of freedom."""
    if tail == "two-sided":
        return float(2.0 * stats.t.sf(np.abs(stat), df))
    elif tail == "greater":
        return float(stats.t.sf(stat, df))
    elif tail == "less":
        return float(stats.t.cdf(stat, df))
    else:
        raise ValueError(f"Unknown tail: {tail}")


class OneSampleTTest(BaseStatisticalTest):
    """One-sample Student's t-test comparing a sample mean against a reference value."""

    def __init__(self, mu0: float = 0.0):
        self.mu0 = float(mu0)

    @property
    def name(self) -> str:
        return "One-Sample t-test"

    def validate_input(self, *data: Any) -> None:
        if len(data) != 1:
            raise ValueError(f"{self.name} expects 1 sample array, got {len(data)}")
        x = np.asarray(data[0], dtype=np.float64)
        if len(x) < 2:
            raise InsufficientDataError(f"{self.name} requires at least 2 observations")
        if not np.isfinite(x).all():
            raise ValueError("Input array contains non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        x = np.asarray(data[0], dtype=np.float64)
        if len(x) >= 3:
            shapiro = stats.shapiro(x)
            norm_ok = bool(shapiro.pvalue >= 0.05)
        else:
            norm_ok = True
        return {"normality": {"passed": norm_ok}}

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        x = np.asarray(data[0], dtype=np.float64)
        n = len(x)
        mean = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        if s <= 1e-12:
            raise InsufficientDataError("Zero variance in sample data")
        df = float(n - 1)
        stat = (mean - self.mu0) / (s / np.sqrt(n))
        p_val = _p_value_from_t(stat, df, hypothesis.tail)
        return float(stat), df, p_val


class IndependentTTest(BaseStatisticalTest):
    """Two-sample Student's t-test assuming equal group variances."""

    @property
    def name(self) -> str:
        return "Independent Two-Sample t-test (Equal Variances)"

    def validate_input(self, *data: Any) -> None:
        if len(data) != 2:
            raise ValueError(f"{self.name} expects 2 sample arrays, got {len(data)}")
        x1, x2 = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
        if len(x1) < 2 or len(x2) < 2:
            raise InsufficientDataError("Each group must contain at least 2 observations")
        if not (np.isfinite(x1).all() and np.isfinite(x2).all()):
            raise ValueError("Groups contain non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        x1, x2 = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
        levene_p = float(stats.levene(x1, x2).pvalue)
        return {
            "equal_variance": {"passed": levene_p >= 0.05, "p_value": levene_p}
        }

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        x1, x2 = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
        n1, n2 = len(x1), len(x2)
        df = float(n1 + n2 - 2)
        s1_sq, s2_sq = np.var(x1, ddof=1), np.var(x2, ddof=1)
        sp = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / df)
        if sp <= 1e-12:
            raise InsufficientDataError("Pooled variance is zero")
        stat = (np.mean(x1) - np.mean(x2)) / (sp * np.sqrt(1.0 / n1 + 1.0 / n2))
        p_val = _p_value_from_t(stat, df, hypothesis.tail)
        return float(stat), df, p_val


class WelchTTest(BaseStatisticalTest):
    """Welch's t-test for independent samples with unequal variances."""

    @property
    def name(self) -> str:
        return "Welch t-test (Unequal Variances)"

    def validate_input(self, *data: Any) -> None:
        if len(data) != 2:
            raise ValueError(f"{self.name} expects 2 sample arrays, got {len(data)}")
        x1, x2 = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
        if len(x1) < 2 or len(x2) < 2:
            raise InsufficientDataError("Each group must contain at least 2 observations")
        if not (np.isfinite(x1).all() and np.isfinite(x2).all()):
            raise ValueError("Groups contain non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        # Welch test does not assume equal variances
        return {"equal_variance_required": False}

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        x1, x2 = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
        n1, n2 = len(x1), len(x2)
        s1_sq, s2_sq = np.var(x1, ddof=1), np.var(x2, ddof=1)
        denom = np.sqrt(s1_sq / n1 + s2_sq / n2)
        if denom <= 1e-12:
            raise InsufficientDataError("Combined standard error is zero")
        stat = (np.mean(x1) - np.mean(x2)) / denom
        # Welch-Satterthwaite degrees of freedom
        df_num = (s1_sq / n1 + s2_sq / n2) ** 2
        df_den = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
        df = float(df_num / df_den)
        p_val = _p_value_from_t(stat, df, hypothesis.tail)
        return float(stat), df, p_val


class PairedTTest(BaseStatisticalTest):
    """Paired Student's t-test comparing paired observations on identical units."""

    @property
    def name(self) -> str:
        return "Paired t-test"

    def validate_input(self, *data: Any) -> None:
        if len(data) == 1:
            diffs = np.asarray(data[0], dtype=np.float64)
        elif len(data) == 2:
            x, y = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
            if len(x) != len(y):
                raise ValueError(f"Paired arrays must have identical length, got {len(x)} and {len(y)}")
            diffs = x - y
        else:
            raise ValueError(f"{self.name} expects either 1 difference array or 2 paired arrays")

        if len(diffs) < 2:
            raise InsufficientDataError(f"{self.name} requires at least 2 pairs")
        if not np.isfinite(diffs).all():
            raise ValueError("Input data contains non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        diffs = data[0] if len(data) == 1 else np.asarray(data[0]) - np.asarray(data[1])
        diffs = np.asarray(diffs, dtype=np.float64)
        if len(diffs) >= 3:
            shapiro = stats.shapiro(diffs)
            norm_ok = bool(shapiro.pvalue >= 0.05)
        else:
            norm_ok = True
        return {"normality_of_differences": {"passed": norm_ok}}

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        diffs = data[0] if len(data) == 1 else np.asarray(data[0]) - np.asarray(data[1])
        diffs = np.asarray(diffs, dtype=np.float64)
        n = len(diffs)
        mean_d = float(np.mean(diffs))
        s_d = float(np.std(diffs, ddof=1))
        if s_d <= 1e-12:
            raise InsufficientDataError("Zero variance in paired differences (all identical)")
        df = float(n - 1)
        stat = mean_d / (s_d / np.sqrt(n))
        p_val = _p_value_from_t(stat, df, hypothesis.tail)
        return float(stat), df, p_val

    def run(self, *data: Any, hypothesis: Hypothesis, unit: str = "patient_group") -> TestResult:
        diffs = data[0] if len(data) == 1 else np.asarray(data[0]) - np.asarray(data[1])
        diffs = np.asarray(diffs, dtype=np.float64)
        n = len(diffs)
        mean_d = float(np.mean(diffs))
        s_d = float(np.std(diffs, ddof=1)) if n > 1 else 1.0
        # Paired Cohen's d_z and Hedges' g
        d_z = mean_d / s_d if s_d > 1e-12 else 0.0
        j_correction = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0)) if n > 2 else 1.0
        hedges_g = d_z * j_correction
        se_d = s_d / np.sqrt(n) if n > 0 else 0.0
        crit_t = float(stats.t.ppf(1.0 - hypothesis.alpha / 2.0, n - 1)) if n > 1 else 1.96
        ci = (float(mean_d - crit_t * se_d), float(mean_d + crit_t * se_d))

        self.validate_input(*data)
        assumptions = self.check_assumptions(*data)
        stat, df, p_val = self.compute(*data, hypothesis=hypothesis)

        return self.interpret(
            statistic=stat,
            df=df,
            p_value=p_val,
            assumptions=assumptions,
            hypothesis=hypothesis,
            n=n,
            effect_size=float(hedges_g),
            ci=ci,
            unit=unit,
        )


class CorrectedResampledTTest(BaseStatisticalTest):
    """Nadeau-Bengio corrected resampled t-test for cross-validation comparisons.

    Corrects the standard t-statistic variance for overlapping training folds:
    t = mean(d) / sqrt((1/K + n_test / n_train) * var(d))
    df = K - 1
    """

    def __init__(self, k_folds: int, n_train: int, n_test: int):
        if k_folds < 2:
            raise ValueError("k_folds must be at least 2")
        if n_train <= 0 or n_test <= 0:
            raise ValueError("n_train and n_test must be positive integers")
        self.k_folds = int(k_folds)
        self.n_train = int(n_train)
        self.n_test = int(n_test)

    @property
    def name(self) -> str:
        return "Nadeau-Bengio Corrected Resampled t-test"

    def validate_input(self, *data: Any) -> None:
        if len(data) != 1:
            raise ValueError(f"{self.name} expects 1 array of K fold differences, got {len(data)}")
        d = np.asarray(data[0], dtype=np.float64)
        if len(d) != self.k_folds:
            raise ValueError(f"Expected {self.k_folds} fold differences, got {len(d)}")
        if not np.isfinite(d).all():
            raise ValueError("Fold differences contain non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        return {
            "resampled_overlap_corrected": True,
            "variance_inflation_factor": float(1.0 / self.k_folds + self.n_test / self.n_train),
        }

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        d = np.asarray(data[0], dtype=np.float64)
        mean_d = float(np.mean(d))
        var_d = float(np.var(d, ddof=1))
        correction_factor = 1.0 / self.k_folds + float(self.n_test) / float(self.n_train)
        corrected_variance = correction_factor * var_d
        if corrected_variance <= 1e-12:
            raise InsufficientDataError("Corrected variance of fold differences is zero")
        stat = mean_d / np.sqrt(corrected_variance)
        df = float(self.k_folds - 1)
        p_val = _p_value_from_t(stat, df, hypothesis.tail)
        return float(stat), df, p_val
