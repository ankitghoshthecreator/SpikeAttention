"""Synthetic and contract integration checks for Beat Dataset v1 artifacts."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_contract_shapes_and_dtypes(tmp_path):
    signals = np.zeros((3, 2, 360), dtype=np.float32)
    labels = np.array([0, 2, 4], dtype=np.int64)
    metadata = pd.DataFrame({"class_id": labels, "patient_group_id": ["p1", "p2", "p3"]})
    path = tmp_path / "fixture.npz"
    np.savez_compressed(path, signals=signals, labels=labels,
                        patient_group_ids=metadata.patient_group_id.to_numpy(dtype="U"),
                        beat_ids=np.array(["a", "b", "c"], dtype="U"),
                        r_peak_indices=np.full(3, 180, dtype=np.int64))
    with np.load(path) as data:
        assert data["signals"].shape == (3, 2, 360)
        assert data["signals"].dtype == np.float32
        assert data["labels"].dtype == np.int64
        assert len(data["patient_group_ids"]) == len(metadata)


def test_processed_beat_dataset_contract_compliance():
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    npz_files = list(processed_dir.glob("beats_*.npz"))
    if not npz_files:
        pytest.skip("Processed dataset not yet generated")

    meta_path = processed_dir / "beats_meta.csv"
    assert meta_path.exists(), "beats_meta.csv must accompany beats_<hash>.npz"

    meta_df = pd.read_csv(meta_path)
    expected_cols = [
        "beat_id", "record_id", "patient_group_id", "annotation_sample",
        "symbol", "class_name", "class_id", "r_peak_index",
        "sampling_rate_hz", "paced_record"
    ]
    assert list(meta_df.columns) == expected_cols

    with np.load(npz_files[0]) as data:
        for key in ["signals", "labels", "patient_group_ids", "beat_ids", "r_peak_indices"]:
            assert key in data, f"Missing key {key} in .npz archive"

        n = len(meta_df)
        assert data["signals"].shape == (n, 2, 360)
        assert data["signals"].dtype == np.float32
        assert data["labels"].shape == (n,)
        assert data["labels"].dtype == np.int64
        assert data["patient_group_ids"].shape == (n,)
        assert data["beat_ids"].shape == (n,)
        assert data["r_peak_indices"].shape == (n,)
        assert data["r_peak_indices"].dtype == np.int64

        assert set(np.unique(data["labels"])).issubset({0, 1, 2, 3, 4})
        assert np.all(data["r_peak_indices"] == 180)
        assert (meta_df["beat_id"].to_numpy(dtype="U") == data["beat_ids"]).all()
        assert (meta_df["class_id"].to_numpy(dtype=np.int64) == data["labels"]).all()
