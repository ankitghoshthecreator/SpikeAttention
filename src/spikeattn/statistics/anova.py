"""Analysis of Variance (ANOVA): One-way and Repeated-Measures ANOVA."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

from spikeattn.exceptions import InsufficientDataError
from spikeattn.statistics.assumption_tests import AssumptionTester
from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


class OneWayANOVA(BaseStatisticalTest):
    """One-Way between-subjects Analysis of Variance."""

    @property
    def name(self) -> str:
        return "One-Way ANOVA"

    def validate_input(self, *groups: Any) -> None:
        if len(groups) < 2:
            raise ValueError("One-Way ANOVA requires at least 2 groups")
        for i, g in enumerate(groups):
            arr = np.asarray(g, dtype=np.float64)
            if len(arr) < 2:
                raise InsufficientDataError(f"Group {i} has fewer than 2 observations")
            if not np.isfinite(arr).all():
                raise ValueError("Group array contains non-finite values")

    def check_assumptions(self, *groups: Any) -> dict[str, Any]:
        valid = [np.asarray(g, dtype=np.float64) for g in groups]
        levene_p = float(stats.levene(*valid).pvalue)
        return {
            "homoscedasticity": {"passed": levene_p >= 0.05, "p_value": levene_p}
        }

    def compute(self, *groups: Any, hypothesis: Hypothesis) -> tuple[float, tuple[float, float], float]:
        valid = [np.asarray(g, dtype=np.float64) for g in groups]
        k = len(valid)
        n_total = sum(len(g) for g in valid)
        df_between = float(k - 1)
        df_within = float(n_total - k)

        res = stats.f_oneway(*valid)
        return float(res.statistic), (df_between, df_within), float(res.pvalue)

    def run(self, *groups: Any, hypothesis: Hypothesis, unit: str = "patient_group") -> TestResult:
        valid = [np.asarray(g, dtype=np.float64) for g in groups]
        self.validate_input(*groups)
        assumptions = self.check_assumptions(*groups)
        stat, df, p_val = self.compute(*groups, hypothesis=hypothesis)

        # Eta-squared = SS_between / SS_total
        grand_mean = np.mean(np.concatenate(valid))
        ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in valid)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in valid)
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        n = sum(len(g) for g in valid)
        return self.interpret(
            statistic=stat,
            df=df,
            p_value=p_val,
            assumptions=assumptions,
            hypothesis=hypothesis,
            n=n,
            effect_size=eta_sq,
            unit=unit,
        )


class RepeatedMeasuresANOVA(BaseStatisticalTest):
    """Repeated-Measures ANOVA across repeated models on identical subjects."""

    @property
    def name(self) -> str:
        return "Repeated-Measures ANOVA"

    def validate_input(self, *data: Any) -> None:
        if len(data) == 1:
            matrix = np.asarray(data[0], dtype=np.float64)
        else:
            matrix = np.column_stack([np.asarray(d, dtype=np.float64) for d in data])

        if matrix.ndim != 2:
            raise ValueError("Repeated-Measures ANOVA requires a 2D matrix [n_subjects, k_models]")
        n_subjects, k_models = matrix.shape
        if n_subjects < 3 or k_models < 2:
            raise InsufficientDataError("RM-ANOVA requires at least 3 subjects and 2 models")
        if not np.isfinite(matrix).all():
            raise ValueError("Data matrix contains non-finite values")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        matrix = np.asarray(data[0], dtype=np.float64) if len(data) == 1 else np.column_stack(data)
        sphericity = AssumptionTester.sphericity(matrix)
        return {"sphericity": sphericity}

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, tuple[float, float], float]:
        matrix = np.asarray(data[0], dtype=np.float64) if len(data) == 1 else np.column_stack(data)
        n_subjects, k_models = matrix.shape

        rows = []
        for s_idx in range(n_subjects):
            for m_idx in range(k_models):
                rows.append({"subject": f"s_{s_idx}", "model": f"m_{m_idx}", "score": matrix[s_idx, m_idx]})
        df_long = pd.DataFrame(rows)

        aov = AnovaRM(df_long, depvar="score", subject="subject", within=["model"])
        res = aov.fit()
        table = res.anova_table

        f_stat = float(table.loc["model", "F Value"])
        df1 = float(table.loc["model", "Num DF"])
        df2 = float(table.loc["model", "Den DF"])
        p_val = float(table.loc["model", "Pr > F"])

        # Greenhouse-Geisser correction if sphericity violated
        sphericity = AssumptionTester.sphericity(matrix)
        eps = sphericity.get("greenhouse_geisser_epsilon", 1.0)
        if eps < 0.75:
            df1_corr = df1 * eps
            df2_corr = df2 * eps
            p_val = float(stats.f.sf(f_stat, df1_corr, df2_corr))
            return f_stat, (df1_corr, df2_corr), p_val

        return f_stat, (df1, df2), p_val

    def run(self, *data: Any, hypothesis: Hypothesis, unit: str = "patient_group") -> TestResult:
        matrix = np.asarray(data[0], dtype=np.float64) if len(data) == 1 else np.column_stack(data)
        n, k = matrix.shape
        self.validate_input(*data)
        assumptions = self.check_assumptions(*data)
        stat, df, p_val = self.compute(*data, hypothesis=hypothesis)

        # Partial eta-squared
        # F = (SS_effect / df1) / (SS_error / df2) => partial_eta_sq = (F * df1) / (F * df1 + df2)
        df1, df2 = df
        partial_eta_sq = float((stat * df1) / (stat * df1 + df2)) if (stat * df1 + df2) > 0 else 0.0

        return self.interpret(
            statistic=stat,
            df=df,
            p_value=p_val,
            assumptions=assumptions,
            hypothesis=hypothesis,
            n=n,
            effect_size=partial_eta_sq,
            unit=unit,
        )
