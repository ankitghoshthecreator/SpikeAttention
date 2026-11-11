"""Unified command line interface for SPIKE-ATTENTION: data, training, analysis, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spikeattn.data.label_mapping import LabelMapper
from spikeattn.data.mitdb_downloader import MitdbDownloader
from spikeattn.data.patient_splitter import PatientGroupCrossValidator
from spikeattn.data.record_catalog import CLASS_ORDER, RecordCatalog
from spikeattn.preprocessing.filters import BandpassFilter
from spikeattn.preprocessing.pipeline import PreprocessingPipeline
from spikeattn.preprocessing.segmentation import BeatSegmenter
from spikeattn.reporting.report_builder import ReportBuilder
from spikeattn.statistics.experiment_analyzer import ExperimentAnalyzer
from spikeattn.utils.config import load_config
from spikeattn.utils.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="spikeattn", description="SPIKE-ATTENTION ECG Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. DATA COMMAND
    data_parser = sub.add_parser("data", help="Data ingestion, cataloging, preprocessing, and splitting")
    actions = data_parser.add_subparsers(dest="action", required=True)
    for action in ("download", "catalog", "preprocess", "split"):
        cmd = actions.add_parser(action)
        cmd.add_argument("--config", default="configs/dataset.yaml")

    # 2. TRAIN COMMAND
    train_parser = sub.add_parser("train", help="Run model training and cross-validation experiments")
    train_parser.add_argument("--run-id", default="run_benchmark", help="Output run identifier")
    train_parser.add_argument("--models", nargs="+", default=None, help="List of models to train: svm cnn rnn plain_snn st_snn")
    train_parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to evaluate")
    train_parser.add_argument("--max-folds", type=int, default=None, help="Max folds per seed to evaluate")
    train_parser.add_argument("--epochs", type=int, default=None, help="Max training epochs per model")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and inference")

    # 3. ANALYZE COMMAND
    analyze_parser = sub.add_parser("analyze", help="Run ISPA statistical analysis on Results-v1 directory")
    analyze_parser.add_argument("--results-dir", required=True, help="Path to results/<run_id> directory")
    analyze_parser.add_argument("--config", default="configs/analysis.yaml", help="Path to analysis.yaml")

    # 4. REPORT COMMAND
    report_parser = sub.add_parser("report", help="Render academic markdown report from analysis bundle")
    report_parser.add_argument("--results-dir", required=True, help="Path to results/<run_id> directory")
    report_parser.add_argument("--config", default="configs/analysis.yaml", help="Path to analysis.yaml")
    report_parser.add_argument("--output", default=None, help="Output markdown path (default: <results-dir>/report.md)")

    args = parser.parse_args()

    if args.command == "data":
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
                pipeline = PreprocessingPipeline(
                    filt, BeatSegmenter(config["window_samples"]), mapper, config["annotators"][0]
                )
                path = pipeline.build(
                    raw_dir,
                    records,
                    config["processed_dir"],
                    exclude_paced_records=config["exclude_paced_records"],
                    paced_records=set(map(str, config["paced_records"])),
                    config_for_hash={
                        "dataset": config,
                        "label_mapping": load_config("configs/label_mapping.yaml", "label_mapping"),
                    },
                )
                print(path)
            else:
                import pandas as pd
                frame = pd.read_csv(Path(config["metadata_dir"]) / "record_catalog.csv")
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

    elif args.command == "train":
        from experiments.run_experiment import run_experiment
        configure_logging()
        run_dir = run_experiment(
            run_id=args.run_id,
            selected_models=args.models,
            seeds_override=args.seeds,
            max_folds=args.max_folds,
            epochs_override=args.epochs,
            batch_size=args.batch_size,
        )
        print(f"Experiment completed. Results stored in: {run_dir}")

    elif args.command == "analyze":
        configure_logging()
        analysis_cfg = load_config(args.config, "analysis")
        analyzer = ExperimentAnalyzer(args.results_dir, analysis_cfg)
        bundle = analyzer.run()
        print(f"Analysis completed successfully across {len(bundle.get('comparisons', []))} model comparisons.")

    elif args.command == "report":
        configure_logging()
        results_dir = Path(args.results_dir)
        analysis_cfg = load_config(args.config, "analysis")
        analyzer = ExperimentAnalyzer(results_dir, analysis_cfg)
        bundle = analyzer.run()
        out_path = Path(args.output) if args.output else results_dir / "report.md"
        builder = ReportBuilder()
        builder.build_report(bundle, out_path, run_id=results_dir.name)
        print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
