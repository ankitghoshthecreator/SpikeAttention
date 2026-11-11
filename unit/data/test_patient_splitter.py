"""Leakage and deterministic split tests."""

import json
from pathlib import Path
import pytest

from spikeattn.data.patient_splitter import PatientGroupCrossValidator, PatientSplitter, assert_disjoint
from spikeattn.data.record_catalog import patient_group_id
from spikeattn.exceptions import InsufficientDataError, LeakageError


def test_201_and_202_share_patient_group():
    assert patient_group_id(201) == patient_group_id(202) == "mitdb:201-202"
    assert patient_group_id("100") == "mitdb:100"


def test_overlap_raises_leakage_error():
    with pytest.raises(LeakageError):
        assert_disjoint(["p1"], ["p1"])
    with pytest.raises(LeakageError):
        assert_disjoint(["p1", "p2"], ["p3"], ["p2", "p4"])


def test_group_cv_is_deterministic_and_disjoint():
    profiles = {f"g{i}": {"N": 10, "V": i % 3} for i in range(20)}
    cv = PatientGroupCrossValidator(5, (17,), 0.2)
    first = [(p.train_patient_group_ids, p.val_patient_group_ids, p.test_patient_group_ids)
             for p, _ in cv.split(profiles, "hash")]
    second = [(p.train_patient_group_ids, p.val_patient_group_ids, p.test_patient_group_ids)
              for p, _ in cv.split(profiles, "hash")]
    assert first == second
    assert len({g for _, _, test in first for g in test}) == len(profiles)


def test_insufficient_groups_raises():
    with pytest.raises(InsufficientDataError):
        list(PatientGroupCrossValidator(5, (0,)).split({"g1": {"N": 10}}, "hash"))
    with pytest.raises(InsufficientDataError):
        PatientSplitter.inner_split(["g1"], {}, seed=0)


def test_generated_splits_disjoint_and_complete():
    splits_dir = Path(__file__).resolve().parents[3] / "data" / "processed" / "splits"
    if not splits_dir.exists():
        pytest.skip("Splits directory not generated yet")
    split_files = list(splits_dir.glob("seed_*_fold_*.json"))
    if not split_files:
        pytest.skip("No split files found")

    paced_groups = {"mitdb:102", "mitdb:104", "mitdb:107", "mitdb:217"}
    all_groups = set()

    for path in split_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0.0"
        train = set(payload["train_patient_group_ids"])
        val = set(payload["val_patient_group_ids"])
        test = set(payload["test_patient_group_ids"])
        assert_disjoint(train, val, test)
        # Excluded paced records must not occur in any partition
        assert train.isdisjoint(paced_groups)
        assert val.isdisjoint(paced_groups)
        assert test.isdisjoint(paced_groups)
        all_groups.update(train | val | test)

    assert len(all_groups) == 43
    for seed in range(5):
        test_union = set()
        for fold in range(5):
            path = splits_dir / f"seed_{seed}_fold_{fold}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            test_union.update(payload["test_patient_group_ids"])
        assert test_union == all_groups
