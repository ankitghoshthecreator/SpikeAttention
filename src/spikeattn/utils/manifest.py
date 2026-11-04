"""Versioned JSON run manifest writer."""

import json
from pathlib import Path
from typing import Any


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a manifest payload as indented UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": "1.0.0", **payload}
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
