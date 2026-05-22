"""Download the PlantVillage dataset.

Source: Kaggle ``abdallahalidev/plantvillage-dataset``.
Role: disease pretraining (PDF §20). ~54k images across 38 classes.
Target: ``data/plant_disease/plantvillage/raw/``.

Idempotent — if the target directory already contains data, the script
logs and exits with status 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `scripts/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    is_directory_populated,
    kaggle_download,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

KAGGLE_SLUG = "abdallahalidev/plantvillage-dataset"
RAW_DIR: Path = DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("PlantVillage already present at %s — skipping.", RAW_DIR)
        return 0
    kaggle_download(KAGGLE_SLUG, RAW_DIR)
    _LOGGER.info("PlantVillage download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
