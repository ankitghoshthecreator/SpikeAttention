"""Algorithmic, pre-specified selection of statistical tests from assumption checks."""

from typing import Any

import numpy as np

from spikeattn.statistics.assumption_tests import AssumptionTester


class TestSelector:
    """Implements pre-registered algorithmic decision rules for hypothesis test selection."""

    @staticmethod
    def select_paired(
        diffs: np.ndarray | list[float],
        alpha_assumption: float = 0.05,
    ) -> dict[str, Any]:
        """Pre-specified rule choosing between Paired t-test and Wilcoxon Signed-Rank test.

        Rule:
        - If Shapiro-Wilk on differences has p >= alpha_assumption AND no extreme outliers exist:
            Select 'paired_t' (parametric)
        - Otherwise:
            Select 'wilcoxon' (non-parametric fallback)
        """
        arr = np.asarray(diffs, dtype=np.float64)
        norm_result = AssumptionTester.shapiro_wilk(arr, alpha=alpha_assumption)
        outlier_result = AssumptionTester.outliers(arr, method="iqr", threshold=3.0)  # 3*IQR for extreme outliers

        norm_ok = bool(norm_result["passed"])
        no_extreme_outliers = bool(not outlier_result["has_outliers"])

        if norm_ok and no_extreme_outliers:
            selected = "paired_t"
            rationale = "Differences show no evidence against normality and no extreme outliers; paired t-test is appropriate."
        else:
            selected = "wilcoxon"
            reasons = []
            if not norm_ok:
                reasons.append(f"normality rejected (p={norm_result['p_value']:.4e})")
            if not no_extreme_outliers:
                reasons.append(f"{outlier_result['count']} extreme outlier(s) detected")
            rationale = f"Parametric assumptions doubtful ({', '.join(reasons)}); selected non-parametric Wilcoxon test."

        return {
            "selected_test": selected,
            "rationale": rationale,
            "normality_p": norm_result["p_value"],
            "normality_passed": norm_ok,
            "extreme_outliers_count": outlier_result["count"],
        }

    @staticmethod
    def select_groups(
        groups: list[np.ndarray],
        alpha_assumption: float = 0.05,
    ) -> dict[str, Any]:
        """Pre-specified rule choosing between Student's t-test and Welch's t-test for two groups."""
        if len(groups) != 2:
            raise ValueError("Group test selection requires exactly 2 groups")

        levene_res = AssumptionTester.levene(groups[0], groups[1], alpha=alpha_assumption)
        equal_var = bool(levene_res["passed"])

        if equal_var:
            selected = "independent_t"
            rationale = "Equal variance assumption not rejected; standard independent two-sample t-test selected."
        else:
            selected = "welch_t"
            rationale = f"Heteroscedasticity detected (p={levene_res['p_value']:.4e}); Welch t-test selected."

        return {
            "selected_test": selected,
            "rationale": rationale,
            "levene_p": levene_res["p_value"],
            "equal_variance_passed": equal_var,
        }
