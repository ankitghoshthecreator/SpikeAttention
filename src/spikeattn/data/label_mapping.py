"""AAMI beat-symbol mapping with explicit non-beat handling."""

from collections import Counter
from pathlib import Path

from spikeattn.exceptions import ConfigError
from spikeattn.utils.config import load_config


class LabelMapper:
    """Map WFDB beat symbols into the five AAMI EC57 classes."""

    def __init__(self, symbol_to_class: dict[str, str], non_beat_symbols: set[str]):
        self.symbol_to_class = symbol_to_class
        self.non_beat_symbols = non_beat_symbols
        covered = set(symbol_to_class) | non_beat_symbols
        if not covered:
            raise ConfigError("Label map is empty")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelMapper":
        """Build mapping from the project label mapping YAML."""
        config = load_config(path, "label_mapping")
        mapping = {symbol: label for label, symbols in config["symbol_to_class"].items() for symbol in symbols}
        if len(mapping) != sum(map(len, config["symbol_to_class"].values())):
            raise ConfigError("A symbol is assigned to more than one class")
        return cls(mapping, set(config.get("non_beat_symbols", [])))

    def map_symbol(self, symbol: str) -> str | None:
        """Return class name, or None for non-beat/unmapped annotations."""
        return self.symbol_to_class.get(symbol)

    def map_annotations(self, symbols: list[str]) -> tuple[list[str | None], dict[str, int]]:
        """Map symbols and count explicit non-beat and unknown annotations."""
        counts: Counter[str] = Counter()
        labels: list[str | None] = []
        for symbol in symbols:
            label = self.map_symbol(symbol)
            labels.append(label)
            if label is None:
                counts["non_beat" if symbol in self.non_beat_symbols else "unmapped"] += 1
        return labels, dict(counts)
