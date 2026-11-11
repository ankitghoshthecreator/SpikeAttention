"""YAML configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml

from spikeattn.exceptions import ConfigError


def validate_config(config: dict[str, Any], kind: str) -> dict[str, Any]:
    """Validate required keys and basic value constraints for a config."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping")
    if kind == "dataset":
        if not isinstance(config.get("dataset"), dict):
            raise ConfigError("dataset config requires a dataset mapping")
        ds = config["dataset"]
        for key in ("raw_dir", "processed_dir", "metadata_dir", "records", "annotators"):
            if key not in ds:
                raise ConfigError(f"dataset config missing {key!r}")
        if not ds["records"] or not all(str(r).isdigit() for r in ds["records"]):
            raise ConfigError("dataset.records must be a non-empty list of record ids")
        if not isinstance(ds.get("window_samples"), int) or ds["window_samples"] < 2:
            raise ConfigError("dataset.window_samples must be an integer >= 2")
        split = ds.get("splits")
        if not isinstance(split, dict) or split.get("n_splits", 0) < 2:
            raise ConfigError("dataset.splits.n_splits must be at least 2")
        if not split.get("seeds") or any(not isinstance(seed, int) or seed < 0 for seed in split["seeds"]):
            raise ConfigError("dataset.splits.seeds must contain non-negative integers")
        fraction = split.get("validation_fraction")
        if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            raise ConfigError("dataset.splits.validation_fraction must be in (0,1)")
        filt = ds.get("filter")
        if not isinstance(filt, dict) or not 0 < filt.get("low_hz", 0) < filt.get("high_hz", 0) < ds.get("sampling_rate_hz", 0) / 2:
            raise ConfigError("dataset.filter must satisfy 0 < low < high < Nyquist")
    elif kind == "label_mapping":
        if config.get("class_order") != ["N", "S", "V", "F", "Q"]:
            raise ConfigError("class_order must be N,S,V,F,Q")
        if not isinstance(config.get("symbol_to_class"), dict):
            raise ConfigError("symbol_to_class must be a mapping")
        if set(config["symbol_to_class"]) != set(config["class_order"]):
            raise ConfigError("symbol_to_class must define exactly the five ordered classes")
        mapped = [symbol for symbols in config["symbol_to_class"].values() for symbol in symbols]
        if len(mapped) != len(set(mapped)):
            raise ConfigError("A symbol may not map to more than one AAMI class")
        if set(mapped) & set(config.get("non_beat_symbols", [])):
            raise ConfigError("A symbol cannot be both a beat class and a non-beat")
    elif kind in {"experiment", "analysis", "efficiency"}:
        if not config:
            raise ConfigError(f"{kind} config may not be empty")
    else:
        raise ConfigError(f"Unknown config kind: {kind}")
    return config


def load_config(path: str | Path, kind: str | None = None) -> dict[str, Any]:
    """Load and validate a YAML mapping.

    Args:
        path: YAML file path.
        kind: Schema selector, inferred from the filename when omitted.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load config {path}: {exc}") from exc
    selected = kind or path.stem
    return validate_config(data, selected)


class ConfigLoader:
    """Small object wrapper for repeated typed config loads."""

    def load(self, path: str | Path, kind: str | None = None) -> dict[str, Any]:
        """Load one YAML config using :func:`load_config`."""
        return load_config(path, kind)
