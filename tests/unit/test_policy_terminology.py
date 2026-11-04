"""Enforce the project-wide ROC terminology policy."""

from pathlib import Path


def test_forbidden_roc_area_token_absent_repo_wide():
    token = "A" + "U" + "C"
    root = Path(__file__).resolve().parents[2]
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if token.casefold() in content.casefold():
            found.append(str(path.relative_to(root)))
    assert not found, f"Disallowed ROC-area terminology found in: {found}"
