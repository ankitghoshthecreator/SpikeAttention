"""Build a filtered, unnormalized beat cache from WFDB records."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from spikeattn.data.label_mapping import LabelMapper
from spikeattn.data.record_catalog import CLASS_ORDER, patient_group_id
from spikeattn.exceptions import DataIntegrityError
from spikeattn.preprocessing.filters import BandpassFilter
from spikeattn.preprocessing.segmentation import BeatSegmenter
from spikeattn.utils.hashing import canonical_hash, sha256_file


class PreprocessingPipeline:
    """Filter full recordings, segment mapped beats, and write Beat Dataset v1."""

    def __init__(self, filter_: BandpassFilter, segmenter: BeatSegmenter, mapper: LabelMapper, annotator: str = "atr"):
        self.filter = filter_
        self.segmenter = segmenter
        self.mapper = mapper
        self.annotator = annotator

    def transform_record(self, base: str | Path, record_id: str | int, paced: bool = False):
        """Return record beats plus metadata; ignore unsupported annotation symbols."""
        record_id = str(record_id).zfill(3)
        try:
            record = wfdb.rdrecord(str(base))
            ann = wfdb.rdann(str(base), self.annotator)
        except Exception as exc:
            raise DataIntegrityError(f"Cannot load {record_id}: {exc}") from exc
        if record.p_signal is None or record.p_signal.shape[1] != 2:
            raise DataIntegrityError(f"Record {record_id} does not have exactly two signal channels")
        filtered = self.filter.transform(record.p_signal)
        labels, _counts = self.mapper.map_annotations(list(ann.symbol))
        paced = "/" in ann.symbol
        keep = [(i, label) for i, label in enumerate(labels) if label is not None]
        indices = np.asarray([int(ann.sample[i]) for i, _ in keep], dtype=np.int64)
        segments, offsets = self.segmenter.segment(filtered, indices)
        metadata = []
        for ordinal, ((annotation_i, label), peak_offset) in enumerate(zip(keep, offsets, strict=True)):
            sample = int(ann.sample[annotation_i])
            metadata.append({
                "beat_id": f"{record_id}:{sample}:{ordinal}", "record_id": record_id,
                "patient_group_id": patient_group_id(record_id), "annotation_sample": sample,
                "symbol": ann.symbol[annotation_i], "class_name": label,
                "class_id": CLASS_ORDER.index(label), "r_peak_index": int(peak_offset),
                "sampling_rate_hz": float(record.fs), "paced_record": bool(paced),
            })
        return segments, metadata, list(record.sig_name), float(record.fs), int(len(ann.symbol))

    def build(self, raw_dir: str | Path, records: list[int | str], output_dir: str | Path,
              dataset_hash: str | None = None, exclude_paced_records: bool = True,
              paced_records: set[str] | None = None, config_for_hash=None) -> Path:
        """Create the compressed beat cache, metadata CSV, and computed facts."""
        raw_dir, output_dir = Path(raw_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paced_records = paced_records or {"102", "104", "107", "217"}
        selected = [str(r).zfill(3) for r in records]
        source_hashes = {}
        for record in selected:
            for path in raw_dir.glob(record + ".*"):
                if path.suffix not in {".dat", ".hea", f".{self.annotator}"}:
                    continue
                source_hashes[path.name] = sha256_file(path)
        dataset_hash = dataset_hash or canonical_hash({"config": config_for_hash, "source_hashes": source_hashes})
        all_signals, rows = [], []
        facts_records = []
        for record in selected:
            is_paced = record in paced_records
            if exclude_paced_records and is_paced:
                continue
            x, meta, leads, hz, annotation_count = self.transform_record(raw_dir / record, record, is_paced)
            all_signals.append(x)
            rows.extend(meta)
            facts_records.append({"record_id": record, "lead_names": leads, "sampling_rate_hz": hz,
                                  "paced_record": is_paced, "annotation_count": annotation_count,
                                  "mapped_beat_count": len(meta)})
        if not all_signals:
            raise DataIntegrityError("No records selected for preprocessing")
        signals = np.concatenate(all_signals).astype(np.float32, copy=False)
        frame = pd.DataFrame(rows)
        path = output_dir / f"beats_{dataset_hash}.npz"
        np.savez_compressed(path, signals=signals,
                            labels=frame.class_id.to_numpy(dtype=np.int64),
                            patient_group_ids=frame.patient_group_id.to_numpy(dtype="U16"),
                            beat_ids=frame.beat_id.to_numpy(dtype="U32"),
                            r_peak_indices=frame.r_peak_index.to_numpy(dtype=np.int64))
        frame.to_csv(output_dir / "beats_meta.csv", index=False)
        counts = {c: int((frame.class_name == c).sum()) for c in CLASS_ORDER}
        facts = {"schema_version": "1.0.0", "dataset": "mitdb", "record_count_processed": len(facts_records),
                 "subject_group_count_processed": int(frame.patient_group_id.nunique()),
                 "sampling_rates_hz": sorted({x["sampling_rate_hz"] for x in facts_records}),
                 "records": facts_records, "paced_records_excluded": sorted(set(selected) & paced_records) if exclude_paced_records else [],
                 "beat_counts_by_aami_class_processed": counts, "source_hashes": source_hashes}
        facts_path = output_dir.parent / "metadata" / "processed_facts.json"
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        facts_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
        return path
