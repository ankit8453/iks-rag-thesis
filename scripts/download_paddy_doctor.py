"""Download the Paddy Doctor dataset.

Source: Kaggle ``petchiammal/paddy-doctor``.
Role: Indian rice disease classification, real-field [ADDED — replaces
§20 Rice Leaf Diseases].
Target: ``data/plant_disease/paddy_doctor/raw/``.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    is_directory_populated,
    kaggle_download,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

KAGGLE_SLUG = "petchiammal/paddy-doctor"
RAW_DIR: Path = DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("Paddy Doctor already present at %s — skipping.", RAW_DIR)
        return 0
    kaggle_download(KAGGLE_SLUG, RAW_DIR)
    _LOGGER.info("Paddy Doctor download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
