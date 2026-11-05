"""Ensure probability metrics use explicit multiclass averaging methodology."""

from pathlib import Path


def test_roc_and_pr_metric_averaging_is_explicit():
    root = Path(__file__).resolve().parents[2]
    content = (root / "src" / "spikeattn" / "evaluation" / "model_evaluator.py").read_text(encoding="utf-8")
    assert "roc_auc_score" in content and "average_precision_score" in content
    assert "macro over classes with both positive and negative examples" in content
