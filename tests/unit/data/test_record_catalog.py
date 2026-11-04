"""Catalog derives record facts from WFDB headers and annotations."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from spikeattn.data.label_mapping import LabelMapper
from spikeattn.data.record_catalog import RecordCatalog


def test_catalog_computes_class_counts_and_group(monkeypatch, tmp_path):
    import spikeattn.data.record_catalog as catalog_module

    monkeypatch.setattr(catalog_module.wfdb, "rdheader", lambda _base: SimpleNamespace(
        sig_name=["MLII", "V1"], fs=360, sig_len=648000))
    monkeypatch.setattr(catalog_module.wfdb, "rdann", lambda _base, _annotator: SimpleNamespace(
        symbol=["N", "V", "+", "/"], sample=[1, 2, 3, 4]))
    mapper = LabelMapper({"N": "N", "V": "V", "/": "Q"}, {"+"})
    catalog = RecordCatalog(tmp_path, [201], mapper)
    frame = catalog.build()
    assert frame.loc[0, "patient_group_id"] == "mitdb:201-202"
    assert frame.loc[0, "beats_N"] == 1
    assert frame.loc[0, "beats_V"] == 1
    assert frame.loc[0, "beats_Q"] == 1
    assert bool(frame.loc[0, "paced_record"])
    path = catalog.write(tmp_path / "record_catalog.csv")
    assert path.exists()
    facts = pd.read_json(tmp_path / "dataset_facts.json", typ="series")
    assert facts["record_count"] == 1
