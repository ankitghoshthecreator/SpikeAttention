"""Integration test: mini end-to-end execution from processed splits to generated report."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_experiment import run_experiment


@pytest.fixture()
def mini_env(tmp_path):
    """Set up a self-contained mini environment with synthetic preprocessed data and splits."""
    data_dir = tmp_path / "data" / "processed"
    splits_dir = data_dir / "splits"
    configs_dir = tmp_path / "configs"
    results_dir = tmp_path / "results"

    data_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    # 1. Synthetic dataset: 40 beats across 4 patient groups (10 beats each)
    N = 40
    rng = np.random.default_rng(42)
    signals = rng.normal(0, 1, (N, 2, 360)).astype(np.float32)
    labels = rng.integers(0, 5, size=N).astype(np.int64)

    patient_groups = ["mitdb:100", "mitdb:101", "mitdb:102", "mitdb:103"]
    meta_rows = []
    for i in range(N):
        pg = patient_groups[i // 10]
        meta_rows.append({
            "beat_id": f"b_{i}",
            "record_id": pg.replace("mitdb:", ""),
            "patient_group_id": pg,
            "label": labels[i],
            "class_id": labels[i],
            "y_true": labels[i],
            "sample_index": i * 360,
        })
    meta_df = pd.DataFrame(meta_rows)

    np.savez(data_dir / "beats_synth123.npz", signals=signals, labels=labels)
    meta_df.to_csv(data_dir / "beats_meta.csv", index=False)

    # 2. Synthetic split: 2 folds (disjoint patient groups)
    # Fold 0: Train [100, 101], Val [102], Test [103]
    split_f0 = {
        "seed": 0,
        "fold": 0,
        "train_patient_group_ids": ["mitdb:100", "mitdb:101"],
        "val_patient_group_ids": ["mitdb:102"],
        "test_patient_group_ids": ["mitdb:103"],
    }
    with open(splits_dir / "seed_0_fold_0.json", "w", encoding="utf-8") as f:
        json.dump(split_f0, f)

    # Fold 1: Train [102, 103], Val [100], Test [101]
    split_f1 = {
        "seed": 0,
        "fold": 1,
        "train_patient_group_ids": ["mitdb:102", "mitdb:103"],
        "val_patient_group_ids": ["mitdb:100"],
        "test_patient_group_ids": ["mitdb:101"],
    }
    with open(splits_dir / "seed_0_fold_1.json", "w", encoding="utf-8") as f:
        json.dump(split_f1, f)

    # 3. Configs
    exp_cfg = {"n_splits": 2, "seeds": [0], "device": "cpu"}
    ds_cfg = {
        "dataset": {
            "name": "mitdb",
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(data_dir),
            "metadata_dir": str(tmp_path / "data" / "metadata"),
            "records": ["100"],
            "annotators": ["atr"],
            "sampling_rate_hz": 360,
            "window_samples": 360,
            "filter": {"low_hz": 0.5, "high_hz": 40.0, "order": 4},
            "splits": {"n_splits": 2, "seeds": [0], "validation_fraction": 0.2},
        }
    }
    analysis_cfg = {"alpha": 0.05, "ci_level": 0.95, "n_bootstrap": 100, "seed": 42}

    import yaml
    with open(configs_dir / "experiment.yaml", "w") as f:
        yaml.dump(exp_cfg, f)
    with open(configs_dir / "dataset.yaml", "w") as f:
        yaml.dump(ds_cfg, f)
    with open(configs_dir / "analysis.yaml", "w") as f:
        yaml.dump(analysis_cfg, f)

    return {
        "configs_dir": configs_dir,
        "data_dir": data_dir,
        "results_dir": results_dir,
    }


class TestEndToEndPipeline:
    def test_mini_pipeline_execution(self, mini_env):
        """Execute mini cross-validation pipeline across SVM and CNN."""
        run_dir = run_experiment(
            config_dir=mini_env["configs_dir"],
            data_dir=mini_env["data_dir"],
            results_root=mini_env["results_dir"],
            run_id="test_mini_run",
            selected_models=["svm", "cnn"],
            seeds_override=[0],
            max_folds=2,
            epochs_override=1,
            batch_size=8,
        )

        assert run_dir.exists()
        assert (run_dir / "predictions" / "test" / "svm_seed0.csv").exists()
        assert (run_dir / "predictions" / "test" / "cnn_seed0.csv").exists()
        assert (run_dir / "predictions" / "validation" / "svm_seed0.csv").exists()
        assert (run_dir / "predictions" / "validation" / "cnn_seed0.csv").exists()
        assert (run_dir / "metrics" / "per_patient.csv").exists()
        assert (run_dir / "metrics" / "per_fold.csv").exists()
        assert (run_dir / "efficiency" / "measurements.csv").exists()
