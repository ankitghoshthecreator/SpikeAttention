"""Non-parametric rank tests: Wilcoxon signed-rank, Friedman, and Kruskal-Wallis."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spikeattn.exceptions import InsufficientDataError
from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


class WilcoxonSignedRankTest(BaseStatisticalTest):
    """Wilcoxon signed-rank test for paired non-parametric comparisons."""

    def __init__(self, zero_method: str = "wilcox"):
        self.zero_method = zero_method

    @property
    def name(self) -> str:
        return "Wilcoxon Signed-Rank Test"

    def validate_input(self, *data: Any) -> None:
        if len(data) == 1:
            diffs = np.asarray(data[0], dtype=np.float64)
        elif len(data) == 2:
            x, y = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
            if len(x) != len(y):
                raise ValueError("Paired arrays must have identical length")
            diffs = x - y
        else:
            raise ValueError(f"{self.name} expects 1 difference array or 2 paired arrays")

        non_zero = diffs[np.abs(diffs) > 1e-12]
        if len(non_zero) < 3:
            raise InsufficientDataError(f"{self.name} requires at least 3 non-zero differences")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        diffs = data[0] if len(data) == 1 else np.asarray(data[0]) - np.asarray(data[1])
        diffs = np.asarray(diffs, dtype=np.float64)
        non_zero = diffs[np.abs(diffs) > 1e-12]
        return {
            "distribution_free": True,
            "symmetric_differences": True,
            "non_zero_pairs": int(len(non_zero)),
        }

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, None, float]:
        diffs = data[0] if len(data) == 1 else np.asarray(data[0]) - np.asarray(data[1])
        diffs = np.asarray(diffs, dtype=np.float64)
        alt = "two-sided" if hypothesis.tail == "two-sided" else hypothesis.tail
        res = stats.wilcoxon(diffs, zero_method=self.zero_method, alternative=alt)
        return float(res.statistic), None, float(res.pvalue)

    def run(self, *data: Any, hypothesis: Hypothesis, unit: str = "patient_group") -> TestResult:
        diffs = data[0] if len(data) == 1 else np.asarray(data[0]) - np.asarray(data[1])
        diffs = np.asarray(diffs, dtype=np.float64)
        non_zero = diffs[np.abs(diffs) > 1e-12]
        n = len(diffs)

        # Compute rank-biserial correlation
        ranks = stats.rankdata(np.abs(non_zero))
        pos_ranks = np.sum(ranks[non_zero > 0])
        neg_ranks = np.sum(ranks[non_zero < 0])
        total_ranks = pos_ranks + neg_ranks
        rank_biserial = float((pos_ranks - neg_ranks) / total_ranks) if total_ranks > 0 else 0.0

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
            effect_size=rank_biserial,
            unit=unit,
        )


class FriedmanTest(BaseStatisticalTest):
    """Friedman test for repeated-measures across K models/conditions."""

    @property
    def name(self) -> str:
        return "Friedman Test"

    def validate_input(self, *data: Any) -> None:
        # Accepts matrix of shape [n_subjects, k_models] or k individual arrays
        if len(data) == 1:
            matrix = np.asarray(data[0], dtype=np.float64)
        else:
            matrix = np.column_stack([np.asarray(d, dtype=np.float64) for d in data])

        if matrix.ndim != 2:
            raise ValueError("Friedman test expects a 2D matrix of shape [n_subjects, k_conditions]")
        n_subjects, k_conditions = matrix.shape
        if n_subjects < 3 or k_conditions < 3:
            raise InsufficientDataError("Friedman test requires at least 3 subjects and 3 conditions")
        if not np.isfinite(matrix).all():
            raise ValueError("Data matrix contains non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        return {
            "non_parametric": True,
            "paired_blocks": True,
        }

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        matrix = np.asarray(data[0], dtype=np.float64) if len(data) == 1 else np.column_stack(data)
        args = [matrix[:, j] for j in range(matrix.shape[1])]
        res = stats.friedmanchisquare(*args)
        df = float(matrix.shape[1] - 1)
        return float(res.statistic), df, float(res.pvalue)

    def run(self, *data: Any, hypothesis: Hypothesis, unit: str = "patient_group") -> TestResult:
        matrix = np.asarray(data[0], dtype=np.float64) if len(data) == 1 else np.column_stack(data)
        n, k = matrix.shape
        self.validate_input(*data)
        assumptions = self.check_assumptions(*data)
        stat, df, p_val = self.compute(*data, hypothesis=hypothesis)

        # Kendall's W = Friedman stat / (n * (k - 1))
        kendalls_w = float(stat / (n * (k - 1))) if (n * (k - 1)) > 0 else 0.0

        return self.interpret(
            statistic=stat,
            df=df,
            p_value=p_val,
            assumptions=assumptions,
            hypothesis=hypothesis,
            n=n,
            effect_size=kendalls_w,
            unit=unit,
        )

    def post_hoc(self, matrix: np.ndarray, model_names: list[str] | None = None, method: str = "holm") -> pd.DataFrame:
        """Perform pairwise post-hoc Wilcoxon signed-rank tests with multiple-comparison correction."""
        mat = np.asarray(matrix, dtype=np.float64)
        n, k = mat.shape
        names = model_names or [f"Model_{j}" for j in range(k)]
        if len(names) != k:
            raise ValueError("Number of names must match matrix columns")

        comparisons = []
        raw_p_values = []
        for i in range(k):
            for j in range(i + 1, k):
                diffs = mat[:, i] - mat[:, j]
                non_zero = diffs[np.abs(diffs) > 1e-12]
                if len(non_zero) < 3:
                    p = 1.0
                    stat = 0.0
                else:
                    res = stats.wilcoxon(diffs, zero_method="wilcox")
                    p = float(res.pvalue)
                    stat = float(res.statistic)
                comparisons.append({
                    "model_a": names[i],
                    "model_b": names[j],
                    "statistic": stat,
                    "raw_p_value": p,
                    "mean_diff": float(np.mean(diffs)),
                })
                raw_p_values.append(p)

        # Apply multiple testing correction
        from spikeattn.statistics.multiple_comparison import MultipleComparisonCorrector
        if method == "holm":
            adj_p = MultipleComparisonCorrector.holm(raw_p_values)
        elif method == "bonferroni":
            adj_p = MultipleComparisonCorrector.bonferroni(raw_p_values)
        else:
            adj_p = MultipleComparisonCorrector.benjamini_hochberg(raw_p_values)

        for comp, ap in zip(comparisons, adj_p, strict=True):
            comp["adjusted_p_value"] = float(ap)
            comp["correction_method"] = method
            comp["reject_h0"] = bool(ap < 0.05)

        return pd.DataFrame(comparisons)


class KruskalWallisTest(BaseStatisticalTest):
    """Kruskal-Wallis one-way ANOVA by ranks for independent groups."""

    @property
    def name(self) -> str:
        return "Kruskal-Wallis H-test"

    def validate_input(self, *groups: Any) -> None:
        if len(groups) < 2:
            raise ValueError(f"{self.name} requires at least 2 independent groups")
        for i, g in enumerate(groups):
            arr = np.asarray(g, dtype=np.float64)
            if len(arr) < 2:
                raise InsufficientDataError(f"Group {i} has fewer than 2 observations")
            if not np.isfinite(arr).all():
                raise ValueError("Group array contains non-finite values")

    def check_assumptions(self, *groups: Any) -> dict[str, Any]:
        return {
            "non_parametric": True,
            "independent_groups": True,
        }

    def compute(self, *groups: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        arrays = [np.asarray(g, dtype=np.float64) for g in groups]
        res = stats.kruskal(*arrays)
        df = float(len(groups) - 1)
        return float(res.statistic), df, float(res.pvalue)
