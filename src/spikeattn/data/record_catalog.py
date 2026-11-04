"""Data-derived MIT-BIH record catalog."""

from pathlib import Path
import json

import pandas as pd
import wfdb

from spikeattn.exceptions import DataIntegrityError
from spikeattn.data.label_mapping import LabelMapper

CLASS_ORDER = ("N", "S", "V", "F", "Q")
def patient_group_id(record_id: str | int) -> str:
    """Return a stable patient group identifier for a record."""
    record = str(record_id).zfill(3)
    return "mitdb:201-202" if record in {"201", "202"} else f"mitdb:{record}"


class RecordCatalog:
    """Build and persist record metadata computed from WFDB source files."""

    patient_group_id = staticmethod(patient_group_id)

    def __init__(self, raw_dir: str | Path, records: list[int | str], mapper: LabelMapper, annotators: list[str] | None = None):
        self.raw_dir = Path(raw_dir)
        self.records = records
        self.mapper = mapper
        self.annotators = annotators or ["atr"]
        self.frame: pd.DataFrame | None = None

    def build(self) -> pd.DataFrame:
        """Read headers and beat annotations, computing counts per class."""
        rows = []
        for item in self.records:
            record = str(item).zfill(3)
            base = self.raw_dir / record
            try:
                header = wfdb.rdheader(str(base))
                annotation = wfdb.rdann(str(base), self.annotators[0])
            except Exception as exc:
                raise DataIntegrityError(f"Cannot read record {record} or annotation: {exc}") from exc
            labels, unmapped = self.mapper.map_annotations(list(annotation.symbol))
            counts = {label: sum(value == label for value in labels) for label in CLASS_ORDER}
            paced_present = "/" in annotation.symbol
            rows.append({
                "record_id": record, "patient_group_id": patient_group_id(record),
                "lead_names": "|".join(header.sig_name), "sampling_rate_hz": float(header.fs),
                "duration_seconds": float(header.sig_len / header.fs),
                "paced_record": paced_present, **{f"beats_{c}": counts[c] for c in CLASS_ORDER},
                "unmapped_annotations": int(unmapped.get("unmapped", 0)),
                "non_beat_annotations": int(unmapped.get("non_beat", 0)),
            })
        self.frame = pd.DataFrame(rows)
        return self.frame.copy()

    def write(self, path: str | Path) -> Path:
        """Write catalog CSV; build it first if needed."""
        if self.frame is None:
            self.build()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(target, index=False)
        facts_path = target.parent / "dataset_facts.json"
        facts = {
            "schema_version": "1.0.0", "dataset": "mitdb",
            "record_count": int(len(self.frame)),
            "subject_group_count": int(self.frame.patient_group_id.nunique()),
            "sampling_rates_hz": sorted(map(float, self.frame.sampling_rate_hz.unique())),
            "durations_seconds": {str(row.record_id): float(row.duration_seconds) for row in self.frame.itertuples()},
            "lead_names_by_record": {str(row.record_id): str(row.lead_names).split("|")
                                     for row in self.frame.itertuples()},
            "records_without_mlii_first_lead": [str(row.record_id) for row in self.frame.itertuples()
                                                if not str(row.lead_names).split("|")[0].upper().startswith("MLII")],
            "paced_record_ids": sorted(self.frame.loc[self.frame.paced_record, "record_id"].astype(str).tolist()),
            "beat_counts_by_class_computed": {
                label: int(self.frame[f"beats_{label}"].sum()) for label in CLASS_ORDER
            },
            "records_201_202_share_group": patient_group_id("201") == patient_group_id("202"),
            "all_records_have_two_channels": all(len(str(x).split("|")) == 2 for x in self.frame.lead_names),
            "all_recordings_are_approximately_half_hour": bool(self.frame.duration_seconds.between(29 * 60, 31 * 60).all()),
            "facts_are_computed_from_downloaded_records": True,
        }
        facts["design_fact_checks"] = {
            "record_count_is_48": facts["record_count"] == 48,
            "subject_group_count_is_47": facts["subject_group_count"] == 47,
            "sampling_rate_is_360_hz": facts["sampling_rates_hz"] == [360.0],
            "102_104_114_lack_mlii_first_lead": set(facts["records_without_mlii_first_lead"]) == {"102", "104", "114"},
            "paced_records_match_design_set": set(facts["paced_record_ids"]) == {"102", "104", "107", "217"},
        }
        facts["design_fact_discrepancies"] = [key for key, passed in facts["design_fact_checks"].items() if not passed]
        facts_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
        return target
