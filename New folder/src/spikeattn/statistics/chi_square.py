"""Chi-Square tests of independence and goodness-of-fit."""

from typing import Any

import numpy as np
from scipy import stats

from spikeattn.exceptions import InsufficientDataError
from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


class ChiSquareTest(BaseStatisticalTest):
    """Chi-Square test of independence on contingency tables and goodness-of-fit."""

    @property
    def name(self) -> str:
        return "Chi-Square Test"

    def validate_input(self, *data: Any) -> None:
        if len(data) != 1:
            raise ValueError(f"{self.name} expects 1 contingency table or observed frequencies")
        table = np.asarray(data[0], dtype=np.float64)
        if np.any(table < 0):
            raise ValueError("Contingency table frequencies must be non-negative")
        if np.sum(table) < 5:
            raise InsufficientDataError("Total sample size in contingency table must be at least 5")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        table = np.asarray(data[0], dtype=np.float64)
        if table.ndim == 2:
            res = stats.contingency.chi2_contingency(table)
            expected = res.expected_freq
            min_exp = float(np.min(expected))
            frac_under_5 = float(np.mean(expected < 5.0))
            return {
                "cochran_rule_passed": bool(min_exp >= 1.0 and frac_under_5 <= 0.20),
                "min_expected_frequency": min_exp,
                "fraction_under_5": frac_under_5,
            }
        else:
            return {"one_dimensional_goodness_of_fit": True}

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float, float]:
        table = np.asarray(data[0], dtype=np.float64)
        if table.ndim == 2:
            res = stats.contingency.chi2_contingency(table, correction=False)
            return float(res.statistic), float(res.dof), float(res.pvalue)
        else:
            res = stats.chisquare(table)
            df = float(len(table) - 1)
            return float(res.statistic), df, float(res.pvalue)

    def independence(self, table: np.ndarray, alpha: float = 0.05, unit: str = "beat") -> TestResult:
        """Run Chi-Square test of independence on a 2D contingency table."""
        tab = np.asarray(table, dtype=np.float64)
        hyp = Hypothesis(
            h0_text="Class distributions are independent across groups",
            h1_text="Class distributions are associated across groups",
            tail="two-sided",
            alpha=alpha,
        )
        self.validate_input(tab)
        assumptions = self.check_assumptions(tab)
        stat, df, p_val = self.compute(tab, hypothesis=hyp)

        # Cramér's V effect size
        n = float(np.sum(tab))
        r, c = tab.shape
        min_dim = min(r - 1, c - 1)
        cramers_v = float(np.sqrt(stat / (n * min_dim))) if (n * min_dim) > 0 else 0.0

        return self.interpret(
            statistic=stat,
            df=df,
            p_value=p_val,
            assumptions=assumptions,
            hypothesis=hyp,
            n=int(n),
            effect_size=cramers_v,
            unit=unit,
        )

    def goodness_of_fit(self, obs: np.ndarray, exp: np.ndarray | None = None, alpha: float = 0.05) -> TestResult:
        """Run Chi-Square goodness-of-fit test against uniform or theoretical proportions."""
        o = np.asarray(obs, dtype=np.float64)
        e = np.asarray(exp, dtype=np.float64) if exp is not None else None
        res = stats.chisquare(o, f_exp=e)
        df = float(len(o) - 1)
        hyp = Hypothesis(
            h0_text="Observed frequencies follow theoretical distribution",
            h1_text="Observed frequencies deviate from theoretical distribution",
            tail="two-sided",
            alpha=alpha,
        )
        return self.interpret(
            statistic=float(res.statistic),
            df=df,
            p_value=float(res.pvalue),
            assumptions={"goodness_of_fit": True},
            hypothesis=hyp,
            n=int(np.sum(o)),
            unit="beat",
        )
