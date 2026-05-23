"""Download the Indian-Region Soil Image Dataset (IRSID — Kaggle mirror).

# TODO[IEEE]:
# This is the Kaggle mirror of IRSID. The full IEEE DataPort version
# (DOI 10.21227/2zz3-f173) includes sieve-analysis ground-truth CSV not
# present in the mirror. If IIITDM institutional IEEE access becomes
# available, replace the Kaggle download with an
# ``ieee_dataport_download(...)`` call (currently a stub in
# ``src/utils/ieee_dataport.py``) and re-run this script.
#
# The sieve-analysis CSV is informational only and must NOT be exposed
# as a model target per master reference §14.

Source: Kaggle ``kiranpandiri/indian-region-soil-image-dataset``.
Role: soil cross-region validation [ADDED — satisfies §14 cross-region
requirement]. 5 texture classes.
Target: ``data/soil/irsid/raw/``.

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

KAGGLE_SLUG = "kiranpandiri/indian-region-soil-image-dataset"
RAW_DIR: Path = DATA_SOIL_DIR / "irsid" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("IRSID already present at %s — skipping.", RAW_DIR)
        return 0
    kaggle_download(KAGGLE_SLUG, RAW_DIR)
    _LOGGER.info("IRSID (Kaggle mirror) download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
