"""Train-only normalizer tests."""

import numpy as np
import pytest

from spikeattn.exceptions import InsufficientDataError, LeakageError
from spikeattn.preprocessing.normalization import LeadNormalizer


def test_fit_records_only_training_groups():
    x = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    groups = np.array(["train-a", "train-b", "test-c", "test-d"])
    normalizer = LeadNormalizer().fit(x, groups, train_groups={"train-a", "train-b"})
    assert normalizer.fit_patient_group_ids == ("train-a", "train-b")
    # Transform on training subset has zero mean and unit variance
    train_transformed = normalizer.transform(x[:2])
    assert np.allclose(train_transformed.mean(axis=(0, 2)), 0, atol=1e-6)
    assert np.allclose(train_transformed.std(axis=(0, 2)), 1, atol=1e-6)


def test_fit_without_explicit_train_groups_raises_leakage():
    x = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    groups = np.array(["train-a", "train-b", "test-c", "test-d"])
    with pytest.raises(LeakageError):
        LeadNormalizer().fit(x, groups)


def test_fit_with_unknown_train_group_raises_leakage():
    x = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    groups = np.array(["train-a", "train-b", "test-c", "test-d"])
    with pytest.raises(LeakageError):
        LeadNormalizer().fit(x, groups, train_groups={"unknown-group"})


def test_unfitted_transform_raises():
    normalizer = LeadNormalizer()
    with pytest.raises(RuntimeError):
        normalizer.transform(np.zeros((2, 2, 10), dtype=np.float32))


def test_zero_variance_raises_insufficient_data():
    x = np.ones((4, 2, 8), dtype=np.float32)
    with pytest.raises(InsufficientDataError):
        LeadNormalizer().fit(x, np.array(["g1", "g2", "g3", "g4"]), train_groups={"g1", "g2"})


def test_invalid_shapes_raise():
    with pytest.raises(InsufficientDataError):
        LeadNormalizer().fit(np.zeros((2, 1, 10)), np.array(["g1", "g2"]), train_groups={"g1"})
    with pytest.raises(InsufficientDataError):
        LeadNormalizer().fit(np.zeros((2, 2, 10)), np.array(["g1"]), train_groups={"g1"})
