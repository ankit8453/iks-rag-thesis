"""Download the OLID I dataset (Bangladesh rice & vegetable leaves).

Source (primary): Kaggle ``raiaone/olid-i`` — single archive, ~14 GB,
4,749 images across 57 multi-label classes. Hosted by the same authors
as the Zenodo record below; the Kaggle packaging is one zip instead of
nineteen, which is materially easier to fetch reliably.

Original paper: Orka et al. (2023), "OLID I: An Open Leaf Image Dataset
for Plant Stress Recognition", *Frontiers in Plant Science* (doi:
10.3389/fpls.2023.1251888).

Role: causation evaluation (C5). The full set is required for the C5
contribution to be evaluable; Phase 4 originally used an 83-image
smoke sample from one crop which was insufficient.

Target: ``data/causation/olid_i/raw/``.

Per master reference §14, the system does NOT infer causation from
images. The OLID multi-label tags here serve as GROUND TRUTH for the
Phase 11 C5 evaluation, not as model targets for the Phase 5 vision
modules.

# FALLBACK (Zenodo, 19 archives) — kept for reference in case Kaggle
# hosting becomes unavailable:
#   ZENODO_RECORD = "8105154"
#   ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
#   FALLBACK_FILES = SMOKE_SAMPLE_FILES or [all 19 keys from the API]
# (See git log entry "fix(Phase 4 §A): Paddy Doctor competition slug
#  + OLID I smoke sample" for the Zenodo-based implementation.)

Idempotent: skip download if ``raw/`` already contains a sentinel
indicating the full dataset (≥4,000 images recursively + the
``class_distribution.xlsx`` metadata file).
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
from src.utils.paths import DATA_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

KAGGLE_SLUG = "raiaone/olid-i"

# Toggle in Phase 11 if/when a smoke sample is needed again; default
# is now the full download per the post-Phase-4 reconciliation.
OLID_FULL_DOWNLOAD: bool = True

CAUSATION_DIR = DATA_DIR / "causation"
RAW_DIR: Path = CAUSATION_DIR / "olid_i" / "raw"

# Sentinel: a populated full OLID has both the class-distribution
# spreadsheet and well over 4,000 images. The Phase-4 smoke sample only
# had 83 images, so the threshold also discriminates against it.
_SENTINEL_FILE = "class_distribution.xlsx"
_MIN_IMAGES_FOR_FULL = 4000


def _looks_like_full_dataset(raw_root: Path) -> bool:
    if not raw_root.is_dir():
        return False
    # Look for the spreadsheet anywhere under raw_root.
    if not any(p.name == _SENTINEL_FILE for p in raw_root.rglob(_SENTINEL_FILE)):
        return False
    image_count = 0
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_exts:
            image_count += 1
            if image_count >= _MIN_IMAGES_FOR_FULL:
                return True
    return False


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if _looks_like_full_dataset(RAW_DIR):
        _LOGGER.info(
            "OLID I full dataset already present at %s — skipping.", RAW_DIR
        )
        return 0

    if not OLID_FULL_DOWNLOAD:
        _LOGGER.warning(
            "OLID_FULL_DOWNLOAD=False but smoke-sample path has been removed "
            "in the Phase-4 fix. Set OLID_FULL_DOWNLOAD=True to proceed."
        )
        return 1

    # Clear any leftover .gitkeep / partial smoke files before the
    # Kaggle archive is unzipped.
    for child in RAW_DIR.iterdir():
        if child.name == ".gitkeep":
            continue
        # Be conservative — only remove obviously-smoke leftovers.
        if child.name in {"class_distribution.xlsx"}:
            child.unlink()
        # Smoke-sample folders like bottle_gourd__DM etc. — leave them
        # to be overwritten by the full archive's contents; Kaggle unzip
        # will silently merge.

    kaggle_download(KAGGLE_SLUG, RAW_DIR)
    _LOGGER.info("OLID I (full Kaggle archive) download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
