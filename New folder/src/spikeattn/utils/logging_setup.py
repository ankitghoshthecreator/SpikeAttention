"""Application logging setup."""

import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure console logging and optional UTF-8 file logging."""
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in root.handlers):
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
    return logging.getLogger("spikeattn")
