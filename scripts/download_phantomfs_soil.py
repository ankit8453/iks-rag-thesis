"""Download the Comprehensive Soil Classification dataset (Phantom-fs).

Source: Kaggle ``ai4a-lab/comprehensive-soil-classification-datasets``.
Role: primary soil-type classification [ADDED — replaces §20 Soil Type
Image Classification]. 7 classes.
Target: ``data/soil/phantomfs/raw/``.

The Kaggle dataset has two variants (original + CyAUG augmented). Phase
4 downloads the original only; CyAUG can be re-pulled in Phase 6 if
Phase 5/6 training needs augmented data.

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
from src.utils.paths import DATA_SOIL_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

KAGGLE_SLUG = "ai4a-lab/comprehensive-soil-classification-datasets"
RAW_DIR: Path = DATA_SOIL_DIR / "phantomfs" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("Phantom-fs soil already present at %s — skipping.", RAW_DIR)
        return 0
    kaggle_download(KAGGLE_SLUG, RAW_DIR)
    _LOGGER.info(
        "Phantom-fs soil download complete. CyAUG variant intentionally "
        "deferred to Phase 6."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
