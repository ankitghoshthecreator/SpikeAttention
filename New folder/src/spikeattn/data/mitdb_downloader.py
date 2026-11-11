"""PhysioNet MIT-BIH downloader and raw record verifier."""

import logging
from pathlib import Path

import wfdb

from spikeattn.exceptions import DataIntegrityError

logger = logging.getLogger(__name__)


class MitdbDownloader:
    """Download records via WFDB's PhysioNet interface."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def download(self, records: list[int | str], annotators: list[str]) -> list[Path]:
        """Fetch requested records and annotations into the raw directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        formatted = [str(r).zfill(3) for r in records]
        try:
            wfdb.dl_database("mitdb", dl_dir=str(self.root), records=formatted, annotators=annotators, keep_subdirs=False, overwrite=False)
            for record in formatted:
                for annotator in annotators:
                    wfdb.rdann(str(self.root / record), annotator)
        except Exception as exc:
            raise DataIntegrityError(
                f"MIT-BIH download failed; retry with `python main.py data download --config configs/dataset.yaml`: {exc}"
            ) from exc
        missing = self.verify(records, annotators)
        if missing:
            raise DataIntegrityError(f"Downloaded MIT-BIH files are incomplete: {missing}")
        return sorted(self.root.glob("*.hea"))

    def verify(self, records: list[int | str], annotators: list[str]) -> list[str]:
        """List missing header, signal, or annotation files."""
        missing: list[str] = []
        for record in records:
            stem = str(self.root / str(record).zfill(3))
            for suffix in (".hea", ".dat"):
                if not Path(stem + suffix).exists():
                    missing.append(stem + suffix)
            for annotator in annotators:
                if not Path(stem + f".{annotator}").exists():
                    missing.append(stem + f".{annotator}")
        return missing
