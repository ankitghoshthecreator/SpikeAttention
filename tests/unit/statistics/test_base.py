"""Unit tests for base.py: Hypothesis, TestResult, and BaseStatisticalTest."""

import pytest

from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult


class TestHypothesis:
    def test_valid_two_sided(self):
        h = Hypothesis(h0_text="No effect", h1_text="Effect exists")
        assert h.tail == "two-sided"
        assert h.alpha == 0.05

    def test_valid_one_sided(self):
        h = Hypothesis(h0_text="H0", h1_text="H1", tail="greater", alpha=0.01)
        assert h.tail == "greater"
        assert h.alpha == 0.01

    def test_invalid_tail_raises(self):
        with pytest.raises(ValueError, match="Invalid tail"):
            Hypothesis(h0_text="H0", h1_text="H1", tail="invalid")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            Hypothesis(h0_text="H0", h1_text="H1", alpha=1.5)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            Hypothesis(h0_text="H0", h1_text="H1", alpha=0.0)

    def test_frozen_immutable(self):
        h = Hypothesis(h0_text="H0", h1_text="H1")
        with pytest.raises((TypeError, AttributeError)):
            h.alpha = 0.01  # type: ignore[misc]


class TestTestResult:
    def _make_result(self, **kwargs):
        defaults = dict(
            name="TestXYZ",
            statistic=2.5,
            df=10.0,
            p_value=0.03,
            alpha=0.05,
            reject_h0=True,
        )
        defaults.update(kwargs)
        return TestResult(**defaults)

    def test_to_dict_has_all_keys(self):
        r = self._make_result()
        d = r.to_dict()
        expected_keys = {
            "name", "statistic", "df", "p_value", "alpha", "reject_h0",
            "critical_region", "effect_size", "ci", "n", "unit",
            "assumptions", "interpretation", "valid",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_values_serialisable(self):
        r = self._make_result(ci=(0.1, 0.9), effect_size=0.6, critical_region=(1.96,))
        d = r.to_dict()
        import json
        json.dumps(d)  # must not raise

    def test_reject_h0_true_when_p_lt_alpha(self):
        r = self._make_result(p_value=0.01, alpha=0.05, reject_h0=True)
        assert r.reject_h0 is True

    def test_reject_h0_false_when_p_ge_alpha(self):
        r = self._make_result(p_value=0.1, alpha=0.05, reject_h0=False)
        assert r.reject_h0 is False

    def test_df_tuple_serialises_to_list(self):
        r = self._make_result(df=(1, 10))
        d = r.to_dict()
        assert isinstance(d["df"], list)

    def test_frozen_immutable(self):
        r = self._make_result()
        with pytest.raises((TypeError, AttributeError)):
            r.p_value = 0.5  # type: ignore[misc]


class _ConcreteTest(BaseStatisticalTest):
    """Minimal concrete implementation for testing the template method."""

    @property
    def name(self) -> str:
        return "ConcreteTest"

    def validate_input(self, *data):
        if len(data[0]) < 2:
            raise ValueError("Need at least 2 samples")

    def check_assumptions(self, *data):
        return {"normality": {"passed": True}}

    def compute(self, *data, hypothesis):
        import numpy as np
        d = np.asarray(data[0])
        stat = float(np.mean(d))
        return stat, len(d) - 1, 0.04


class TestBaseStatisticalTest:
    def test_run_returns_test_result(self):
        import numpy as np
        hyp = Hypothesis(h0_text="H0", h1_text="H1")
        result = _ConcreteTest().run(np.arange(1, 11, dtype=float), hypothesis=hyp)
        assert isinstance(result, TestResult)

    def test_run_calls_all_steps(self):
        import numpy as np
        hyp = Hypothesis(h0_text="H0", h1_text="H1", alpha=0.05)
        r = _ConcreteTest().run(np.arange(5, dtype=float), hypothesis=hyp)
        assert r.name == "ConcreteTest"
        assert r.p_value == pytest.approx(0.04)

    def test_run_raises_on_invalid_input(self):
        import numpy as np
        hyp = Hypothesis(h0_text="H0", h1_text="H1")
        with pytest.raises(ValueError):
            _ConcreteTest().run(np.array([1.0]), hypothesis=hyp)

    def test_interpret_reject_h0_text(self):
        import numpy as np
        hyp = Hypothesis(h0_text="H0 holds", h1_text="H1 holds", alpha=0.05)
        r = _ConcreteTest().run(np.arange(10, dtype=float), hypothesis=hyp)
        # p=0.04 < 0.05 → reject
        assert "Reject H0" in r.interpretation
        assert "H1 holds" in r.interpretation
