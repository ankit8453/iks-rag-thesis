"""Download the PlantDoc dataset (cropped variant).

Source: GitHub ``pratikkayal/PlantDoc-Dataset``.
Role: disease real-field evaluation (PDF §20). ~2.5k images, 27 classes.
Target: ``data/plant_disease/plantdoc/raw/``.

The repository contains both the original variant and a ``cropped``
variant; we use the cropped variant for the disease evaluation.

Idempotent.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    git_clone,
    is_directory_populated,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
RAW_DIR: Path = DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR.parent)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("PlantDoc already present at %s — skipping.", RAW_DIR)
        return 0

    # git refuses to clone into a populated directory, so clone into a
    # sibling temp dir and then move the contents we actually want.
    temp_clone = RAW_DIR.parent / "_plantdoc_repo"
    shutil.rmtree(temp_clone, ignore_errors=True)
    git_clone(REPO_URL, temp_clone)

    # PlantDoc layout (as of the upstream repo): the cropped variant lives
    # under a directory named ``cropped`` containing per-class folders.
    # Some forks instead nest it as ``PlantDoc-Dataset/cropped``. Search.
    cropped_candidates = list(temp_clone.rglob("cropped"))
    cropped_dirs = [c for c in cropped_candidates if c.is_dir()]
    if not cropped_dirs:
        _LOGGER.error(
            "Could not find a 'cropped' directory inside the PlantDoc clone "
            "at %s. Leaving the clone in place for manual inspection.",
            temp_clone,
        )
        return 1

    cropped_dir = cropped_dirs[0]
    _LOGGER.info("Using cropped variant at %s", cropped_dir)
    ensure_raw_dir(RAW_DIR)
    # Move each per-class subdir under RAW_DIR.
    for child in cropped_dir.iterdir():
        shutil.move(str(child), str(RAW_DIR / child.name))

    # Clean up the rest of the clone — we only need the cropped class
    # folders.
    shutil.rmtree(temp_clone, ignore_errors=True)
    _LOGGER.info("PlantDoc cropped variant ready at %s", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
