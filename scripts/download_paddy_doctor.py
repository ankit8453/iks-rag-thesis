"""Download the Paddy Doctor dataset.

Source: Kaggle **competition** ``paddy-disease-classification``
(the original slug ``petchiammal/paddy-doctor`` in the prompt was
incorrect — the canonical Paddy Doctor data lives as a competition,
not a dataset). The competition variant ships the subset of 10,407
images used in the IEEE benchmarking paper.

Role: Indian rice disease classification, real-field [ADDED — replaces
§20 Rice Leaf Diseases].
Target: ``data/plant_disease/paddy_doctor/raw/``.

Before this script will succeed the Kaggle account must have joined
the competition on the web UI
(https://www.kaggle.com/competitions/paddy-disease-classification);
otherwise the API call returns ``403 Forbidden``.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    is_directory_populated,
    kaggle_competition_download,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

KAGGLE_COMPETITION = "paddy-disease-classification"
RAW_DIR: Path = DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("Paddy Doctor already present at %s — skipping.", RAW_DIR)
        return 0
    kaggle_competition_download(KAGGLE_COMPETITION, RAW_DIR)
    _LOGGER.info("Paddy Doctor (competition variant) download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
