"""Base classes, dataclasses, and template methods for ISPA statistical testing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Hypothesis:
    """Formal hypothesis declaration for statistical testing.

    Attributes:
        h0_text: Null hypothesis description in plain text.
        h1_text: Alternative hypothesis description in plain text.
        tail: Direction of the test ('two-sided', 'greater', 'less').
        alpha: Significance threshold level (default 0.05).
    """
    h0_text: str
    h1_text: str
    tail: str = "two-sided"
    alpha: float = 0.05

    def __post_init__(self):
        if self.tail not in {"two-sided", "greater", "less"}:
            raise ValueError(f"Invalid tail '{self.tail}'; must be 'two-sided', 'greater', or 'less'")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {self.alpha}")


@dataclass(frozen=True)
class TestResult:
    """Immutable encapsulation of a statistical test outcome.

    Attributes:
        name: Name of the statistical test.
        statistic: Computed test statistic value.
        df: Degrees of freedom (scalar, tuple, or None).
        p_value: Computed p-value.
        alpha: Significance threshold used for the decision.
        reject_h0: Boolean outcome (True if p_value < alpha).
        critical_region: Bounds of the rejection critical region, if applicable.
        effect_size: Associated effect size value, if applicable.
        ci: Confidence interval for the test parameter (lower, upper), if computed.
        n: Sample size (number of observations/pairs).
        unit: Statistical observation unit ('patient_group', 'fold', 'seed', 'beat').
        assumptions: Dictionary of assumption check results.
        interpretation: Plain-language scientific interpretation.
        valid: Whether test assumptions and inputs were satisfied.
    """
    name: str
    statistic: float
    df: float | tuple[float, ...] | None
    p_value: float
    alpha: float
    reject_h0: bool
    critical_region: tuple[float, ...] | None = None
    effect_size: float | None = None
    ci: tuple[float, float] | None = None
    n: int = 0
    unit: str = "patient_group"
    assumptions: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "name": self.name,
            "statistic": float(self.statistic),
            "df": self.df if self.df is None or isinstance(self.df, (int, float)) else list(self.df),
            "p_value": float(self.p_value),
            "alpha": float(self.alpha),
            "reject_h0": bool(self.reject_h0),
            "critical_region": list(self.critical_region) if self.critical_region else None,
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "ci": list(self.ci) if self.ci is not None else None,
            "n": int(self.n),
            "unit": self.unit,
            "assumptions": self.assumptions,
            "interpretation": self.interpretation,
            "valid": bool(self.valid),
        }


TestResult.__test__ = False


class BaseStatisticalTest(ABC):
    """Abstract base class employing the Template Method pattern for hypothesis tests."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the statistical test."""
        pass

    @abstractmethod
    def validate_input(self, *data: Any) -> None:
        """Validate shapes, types, and sample sizes of input data."""
        pass

    @abstractmethod
    def check_assumptions(self, *data: Any) -> dict[str, Any]:
        """Check mathematical and distributional assumptions."""
        pass

    @abstractmethod
    def compute(self, *data: Any, hypothesis: Hypothesis) -> tuple[float, float | tuple[float, ...] | None, float]:
        """Compute the test statistic, degrees of freedom, and p-value.

        Returns:
            (statistic, df, p_value)
        """
        pass

    def interpret(
        self,
        statistic: float,
        df: float | tuple[float, ...] | None,
        p_value: float,
        assumptions: dict[str, Any],
        hypothesis: Hypothesis,
        n: int,
        effect_size: float | None = None,
        ci: tuple[float, float] | None = None,
        unit: str = "patient_group",
    ) -> TestResult:
        """Synthesize test results into an immutable TestResult with plain-language text."""
        reject = bool(p_value < hypothesis.alpha)
        decision_str = "Reject H0" if reject else "Fail to reject H0"
        df_str = f", df={df}" if df is not None else ""
        effect_str = f", effect size={effect_size:.4f}" if effect_size is not None else ""
        ci_str = f", 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else ""

        text = (
            f"{self.name} on unit '{unit}' (N={n}): "
            f"statistic={statistic:.4f}{df_str}, p-value={p_value:.4e} (alpha={hypothesis.alpha}). "
            f"{decision_str}. {hypothesis.h1_text if reject else hypothesis.h0_text}{effect_str}{ci_str}."
        )

        all_assumptions_met = all(
            v.get("passed", True) if isinstance(v, dict) else bool(v)
            for v in assumptions.values()
        )

        return TestResult(
            name=self.name,
            statistic=float(statistic),
            df=df,
            p_value=float(p_value),
            alpha=float(hypothesis.alpha),
            reject_h0=reject,
            effect_size=effect_size,
            ci=ci,
            n=n,
            unit=unit,
            assumptions=assumptions,
            interpretation=text,
            valid=all_assumptions_met,
        )

    def run(self, *data: Any, hypothesis: Hypothesis, unit: str = "patient_group") -> TestResult:
        """Template method executing the canonical hypothesis testing pipeline."""
        self.validate_input(*data)
        assumptions = self.check_assumptions(*data)
        statistic, df, p_value = self.compute(*data, hypothesis=hypothesis)
        n = len(data[0]) if len(data) > 0 and hasattr(data[0], "__len__") else 0
        return self.interpret(
            statistic=statistic,
            df=df,
            p_value=p_value,
            assumptions=assumptions,
            hypothesis=hypothesis,
            n=n,
            unit=unit,
        )
