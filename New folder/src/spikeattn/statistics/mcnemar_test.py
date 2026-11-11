"""McNemar's test for paired classification discordance."""

from typing import Any

import numpy as np
from scipy import stats

from spikeattn.exceptions import InsufficientDataError
from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


class McNemarTest(BaseStatisticalTest):
    """McNemar's test on paired binary correctness arrays.

    Caveat: When applied to pooled beats across recordings, beats within the same
    patient are correlated, violating independent observations. Results on pooled beats
    must be flagged as exploratory.
    """

    @property
    def name(self) -> str:
        return "McNemar's Test"

    def validate_input(self, *data: Any) -> None:
        if len(data) != 2:
            raise ValueError(f"{self.name} expects 2 binary correctness arrays (model_a, model_b)")
        a, b = np.asarray(data[0], dtype=bool), np.asarray(data[1], dtype=bool)
        if len(a) != len(b):
            raise ValueError(f"Arrays must have identical length, got {len(a)} and {len(b)}")
        if len(a) < 10:
            raise InsufficientDataError(f"{self.name} requires at least 10 observations")

    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        a, b = np.asarray(data[0], dtype=bool), np.asarray(data[1], dtype=bool)
        b_count = int(np.sum(a & (~b)))  # Model A correct, Model B incorrect
        c_count = int(np.sum((~a) & b))  # Model A incorrect, Model B correct
        discordant = b_count + c_count
        return {
            "discordant_pairs": discordant,
            "b_count": b_count,
            "c_count": c_count,
            "exact_binomial_used": bool(discordant < 25),
            "beat_level_pseudoreplication_warning": "Observations may be clustered within patients; interpret with caution",
        }

    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float | None, float]:
        a, b = np.asarray(data[0], dtype=bool), np.asarray(data[1], dtype=bool)
        b_count = int(np.sum(a & (~b)))
        c_count = int(np.sum((~a) & b))
        discordant = b_count + c_count

        if discordant == 0:
            return 0.0, None, 1.0

        if discordant < 25:
            # Exact two-sided binomial test
            k = min(b_count, c_count)
            res = stats.binomtest(k, n=discordant, p=0.5, alternative="two-sided")
            return float(k), None, float(res.pvalue)
        else:
            # Chi-square with Edwards continuity correction
            stat = float(((abs(b_count - c_count) - 1.0) ** 2) / discordant)
            df = 1.0
            p_val = float(stats.chi2.sf(stat, df))
            return stat, df, p_val

    def run(self, *data: Any, hypothesis: Hypothesis, unit: str = "beat") -> TestResult:
        a, b = np.asarray(data[0], dtype=bool), np.asarray(data[1], dtype=bool)
        n = len(a)
        self.validate_input(*data)
        assumptions = self.check_assumptions(*data)
        stat, df, p_val = self.compute(*data, hypothesis=hypothesis)

        b_count = assumptions["b_count"]
        c_count = assumptions["c_count"]
        odds_ratio = float(b_count / c_count) if c_count > 0 else float("inf")

        return self.interpret(
            statistic=stat,
            df=df,
            p_value=p_val,
            assumptions=assumptions,
            hypothesis=hypothesis,
            n=n,
            effect_size=odds_ratio,
            unit=unit,
        )
