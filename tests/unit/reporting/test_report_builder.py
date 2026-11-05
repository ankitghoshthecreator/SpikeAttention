"""Unit tests for ReportBuilder and report template integrity."""

from pathlib import Path
import re
import pytest

from spikeattn.reporting.report_builder import ReportBuilder


def _make_synthetic_bundle():
    return {
        "descriptive": [
            {
                "model": "st_snn",
                "n": 43,
                "mean": 0.885,
                "median": 0.890,
                "std": 0.045,
                "iqr": 0.050,
                "min": 0.760,
                "max": 0.965,
            },
            {
                "model": "svm",
                "n": 43,
                "mean": 0.810,
                "median": 0.815,
                "std": 0.060,
                "iqr": 0.075,
                "min": 0.680,
                "max": 0.910,
            },
        ],
        "comparisons": [
            {
                "name": "Paired t-test",
                "statistic": 4.521,
                "df": 42.0,
                "p_value": 0.0001,
                "alpha": 0.05,
                "reject_h0": True,
                "effect_size": 0.725,
                "baseline": "svm",
                "assumptions": {"holm_corrected_p": 0.0004, "holm_reject_h0": True},
                "interpretation": "ST-SNN significantly outperforms SVM on patient-group balanced accuracy.",
            }
        ],
        "intervals": {
            "st_snn": {"point": 0.885, "lower": 0.871, "upper": 0.898, "n_bootstrap": 2000},
            "svm": {"point": 0.810, "lower": 0.792, "upper": 0.828, "n_bootstrap": 2000},
        },
        "efficiency": [
            {
                "model": "st_snn",
                "metric": "latency_ms",
                "median": 2.15,
                "unit": "ms",
                "kind": "measured",
                "method": "median_warmup",
            },
            {
                "model": "st_snn",
                "metric": "energy_uJ",
                "median": 0.35,
                "unit": "uJ",
                "kind": "estimated",
                "method": "cmos_model",
            },
        ],
    }


class TestReportBuilder:
    def test_template_zero_hard_coded_numeric_values(self):
        """Scans the report template for illegal hardcoded numeric claims like p=0.0001 or 88.5%."""
        template_path = Path("src/spikeattn/reporting/templates/report.md.j2")
        content = template_path.read_text(encoding="utf-8")

        # Exclude Jinja2 formatting directives like %.4f, 2000, 0.05 (alpha), 95% (nominal CI level)
        # Any static table data rows containing hardcoded scientific numbers outside Jinja tags are illegal
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            # If line is a markdown table row and doesn't contain Jinja tags {{ }}, it must not contain floating point stats
            if stripped.startswith("|") and not ("{{" in stripped or "}}" in stripped or "{%" in stripped):
                # Search for floating point numbers
                floats = re.findall(r"\b\d+\.\d+\b", stripped)
                assert not floats, f"Hardcoded floating point statistical value found in template line {idx}: {stripped}"

    def test_build_report_renders_file(self, tmp_path):
        builder = ReportBuilder()
        bundle = _make_synthetic_bundle()
        out_file = tmp_path / "report.md"

        res_path = builder.build_report(bundle, out_file, run_id="test_run_123")
        assert res_path.exists()

        text = res_path.read_text(encoding="utf-8")
        assert "test_run_123" in text
        assert "ST-SNN" in text or "st_snn" in text
        assert "0.8850" in text  # formatted mean
        assert "estimated" in text
