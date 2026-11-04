"""Small Part 1 command line entry point."""

import argparse
import json
from pathlib import Path

from spikeattn.data.label_mapping import LabelMapper
from spikeattn.data.mitdb_downloader import MitdbDownloader
from spikeattn.data.patient_splitter import PatientGroupCrossValidator
from spikeattn.data.record_catalog import RecordCatalog, CLASS_ORDER
from spikeattn.preprocessing.filters import BandpassFilter
from spikeattn.preprocessing.pipeline import PreprocessingPipeline
from spikeattn.preprocessing.segmentation import BeatSegmenter
from spikeattn.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(prog="spikeattn")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data")
    actions = data.add_subparsers(dest="action", required=True)
    for action in ("download", "catalog", "preprocess", "split"):
        cmd = actions.add_parser(action)
        cmd.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()
    config = load_config(args.config, "dataset")["dataset"]
    records = config["records"]
    raw_dir = Path(config["raw_dir"])
    if args.action == "download":
        paths = MitdbDownloader(raw_dir).download(records, config["annotators"])
        print(json.dumps({"downloaded_header_count": len(paths)}))
    else:
        mapper = LabelMapper.from_yaml("configs/label_mapping.yaml")
        if args.action == "catalog":
            catalog = RecordCatalog(raw_dir, records, mapper, config["annotators"])
            catalog.write(Path(config["metadata_dir"]) / "record_catalog.csv")
            print("record catalog written")
        elif args.action == "preprocess":
            filt = BandpassFilter(config["sampling_rate_hz"], **config["filter"])
            pipeline = PreprocessingPipeline(filt, BeatSegmenter(config["window_samples"]), mapper,
                                             config["annotators"][0])
            path = pipeline.build(raw_dir, records, config["processed_dir"],
                                  exclude_paced_records=config["exclude_paced_records"],
                                  paced_records=set(map(str, config["paced_records"])),
                                  config_for_hash={"dataset": config,
                                                   "label_mapping": load_config("configs/label_mapping.yaml", "label_mapping")})
            print(path)
        else:
            frame = __import__("pandas").read_csv(Path(config["metadata_dir"]) / "record_catalog.csv")
            if config["exclude_paced_records"]:
                frame = frame.loc[~frame.paced_record.astype(bool)]
            profiles = {}
            for group, rows in frame.groupby("patient_group_id"):
                profiles[str(group)] = {c: int(rows[f"beats_{c}"].sum()) for c in CLASS_ORDER}
            validator = PatientGroupCrossValidator(**config["splits"])
            cache_files = list(Path(config["processed_dir"]).glob("beats_*.npz"))
            if len(cache_files) != 1:
                raise RuntimeError("Expected exactly one preprocessed beat cache before creating splits")
            dataset_hash = cache_files[0].stem.removeprefix("beats_")
            paths = validator.write(profiles, Path(config["processed_dir"]) / "splits", dataset_hash)
            print(json.dumps({"split_files": len(paths)}))


if __name__ == "__main__":
    main()
