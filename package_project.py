"""Create ENDOCARE.zip beside the project folder, excluding caches and the dataset CSV."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT.parent / "ENDOCARE.zip"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".ruff_cache",
    "catboost_info",
}
SKIP_FILE_SUFFIXES = {".pyc", ".pyo", ".log"}
SKIP_FILE_NAMES = {"structured_endometriosis_data.csv"}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIR_NAMES:
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    if path.suffix in SKIP_FILE_SUFFIXES:
        return True
    return False


def main() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path.relative_to(ROOT)):
                continue
            arcname = Path("ENDOCARE") / path.relative_to(ROOT)
            zf.write(path, arcname.as_posix())
            count += 1
    print(f"Wrote {ZIP_PATH} with {count} files (dataset CSV excluded).")


if __name__ == "__main__":
    main()
