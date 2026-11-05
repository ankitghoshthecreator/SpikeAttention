import pandas as pd
import pytest

from spikeattn.visualization.performance_plots import (
    ARRHYTHMIA_CLASS_ORDER,
    plot_evaluation_comparisons,
)


def _frames(models=("cnn",), classes=("N", "S", "V", "F", "Q")):
    metrics = pd.DataFrame([
        {"model": model, "balanced_accuracy": 0.5, "macro_f1": 0.4}
        for model in models
    ])
    per_class = pd.DataFrame([
        {"model": model, "class": cls, "precision": .4, "recall": .5, "f1": .45}
        for model in models for cls in classes
    ])
    return metrics, per_class


def test_single_model_per_class_plotting(tmp_path):
    metrics, per_class = _frames()
    outputs = plot_evaluation_comparisons(metrics, per_class, {}, tmp_path, "test")
    assert (tmp_path / "per_class_metrics.png").exists()
    assert (tmp_path / "per_class_f1.png").exists()
    assert outputs


def test_multi_model_per_class_plotting(tmp_path):
    metrics, per_class = _frames(("cnn", "rnn", "st_snn"))
    plot_evaluation_comparisons(metrics, per_class, {}, tmp_path, "test")
    assert (tmp_path / "per_class_metrics.png").exists()


def test_missing_classes_keep_canonical_order_and_plot(tmp_path):
    metrics, per_class = _frames(classes=("V", "N", "F"))
    plot_evaluation_comparisons(metrics, per_class, {}, tmp_path, "test")
    assert ARRHYTHMIA_CLASS_ORDER == ["N", "S", "V", "F", "Q"]
    assert (tmp_path / "per_class_f1.png").exists()


def test_empty_per_class_input_has_clear_error(tmp_path):
    metrics, per_class = _frames()
    with pytest.raises(ValueError, match="per_class metrics must be a non-empty DataFrame"):
        plot_evaluation_comparisons(metrics, per_class.iloc[0:0], {}, tmp_path, "test")


def test_missing_model_identifier_has_clear_error(tmp_path):
    metrics, per_class = _frames()
    per_class = per_class.drop(columns="model")
    with pytest.raises(ValueError) as exc:
        plot_evaluation_comparisons(metrics, per_class, {}, tmp_path, "test")
    assert "per_class metrics is missing required columns: ['model']" in str(exc.value)


def test_model_identifiers_must_match_between_frames(tmp_path):
    metrics, per_class = _frames()
    per_class["model"] = "rnn"
    with pytest.raises(ValueError, match="model identifiers differ"):
        plot_evaluation_comparisons(metrics, per_class, {}, tmp_path, "test")
