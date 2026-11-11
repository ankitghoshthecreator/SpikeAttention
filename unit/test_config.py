"""Config schema tests."""

from pathlib import Path

import pytest

from spikeattn.exceptions import ConfigError
from spikeattn.utils.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["dataset", "label_mapping", "experiment", "analysis", "efficiency"])
def test_repository_configs_load(name):
    assert load_config(ROOT / "configs" / f"{name}.yaml", name)


def test_invalid_class_order_raises():
    with pytest.raises(ConfigError):
        validate_config({"class_order": ["V"], "symbol_to_class": {}}, "label_mapping")
