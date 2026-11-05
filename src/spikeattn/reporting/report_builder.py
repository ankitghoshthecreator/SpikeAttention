"""Academic report generation from empirical AnalysisBundle using Jinja2."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


class ReportBuilder:
    """Renders comprehensive academic Markdown reports from empirical AnalysisBundle dictionaries."""

    def __init__(self, template_dir: Path | str | None = None) -> None:
        if template_dir is None:
            self.template_dir = Path(__file__).resolve().parent / "templates"
        else:
            self.template_dir = Path(template_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build_report(
        self,
        analysis_bundle: dict[str, Any],
        output_path: Path | str,
        run_id: str = "run_ispa",
        validation_metrics: list[dict[str, Any]] | None = None,
        test_metrics: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Render the complete report document to disk.

        Args:
            analysis_bundle: Dictionary produced by ExperimentAnalyzer.report().
            output_path: Destination path for report.md.
            run_id: Identifier of the experimental run.

        Returns:
            Path of written report.
        """
        template = self.env.get_template("report.md.j2")

        context = dict(analysis_bundle)
        context.setdefault("paired_difference_intervals", {})
        context.setdefault("posthoc", [])
        context.setdefault("friedman", None)
        context["run_id"] = run_id
        context["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        context["validation_metrics"] = validation_metrics or []
        context["test_metrics"] = test_metrics or []

        # Normalize comparisons for template iteration if baseline is not explicitly keyed
        comparisons_norm = []
        raw_comps = context.get("comparisons", [])
        baselines = ["svm", "cnn", "rnn", "plain_snn"]

        for idx, comp in enumerate(raw_comps):
            comp_dict = dict(comp)
            if "baseline" not in comp_dict:
                comp_dict["baseline"] = baselines[idx] if idx < len(baselines) else f"baseline_{idx}"
            comparisons_norm.append(comp_dict)
        context["comparisons"] = comparisons_norm

        content = template.render(**context)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return out
