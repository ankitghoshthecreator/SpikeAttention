"""Label mapping coverage and output tests."""

from pathlib import Path

from spikeattn.data.label_mapping import LabelMapper


def test_map_coverage_and_nonbeat():
    root = Path(__file__).resolve().parents[3]
    mapper = LabelMapper.from_yaml(root / "configs" / "label_mapping.yaml")
    config_symbols = [symbol for symbols in mapper.symbol_to_class.values() for symbol in symbols]
    labels, counts = mapper.map_annotations(config_symbols + ["+", "?"])
    assert all(label is not None for label in labels[:len(config_symbols)])
    assert counts == {"non_beat": 1, "unmapped": 1}
    assert set(mapper.symbol_to_class.values()) == {"N", "S", "V", "F", "Q"}
