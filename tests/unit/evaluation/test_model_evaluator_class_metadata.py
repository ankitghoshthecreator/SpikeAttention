import numpy as np

from spikeattn.evaluation.model_evaluator import ModelEvaluator
from spikeattn.prediction.predictor import PredictionBundle


def test_classification_evaluation_propagates_model_identifier():
    bundle = PredictionBundle(
        beat_ids=[f"b{i}" for i in range(5)],
        patient_group_ids=["p1"] * 5,
        record_ids=["r1"] * 5,
        seed=0,
        fold=0,
        model_name="cnn",
        y_true=np.array([0, 0, 1, 1, 3]),
        y_pred=np.array([0, 1, 1, 1, 3]),
        probabilities=np.array([
            [.8, .1, .04, .03, .03],
            [.2, .7, .04, .03, .03],
            [.1, .8, .04, .03, .03],
            [.1, .8, .04, .03, .03],
            [.1, .1, .04, .7, .03],
        ]),
    )
    summary, per_class, _, curves = ModelEvaluator.compute_classification(bundle)
    assert summary["model"] == "cnn"
    assert per_class["model"].unique().tolist() == ["cnn"]
    assert per_class["class"].tolist() == ["N", "S", "V", "F", "Q"]
    assert per_class.loc[per_class["class"].isin(["V", "Q"]), "support"].tolist() == [0, 0]
    assert curves["classes"]["V"]["status"] == "undefined"
