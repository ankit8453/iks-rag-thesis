"""Download the OLID I dataset (Bangladesh rice leaves, multi-label).

Source: Zenodo record 8105154, DOI ``10.5281/zenodo.8105154``.
Role: causation evaluation (C5) [ADDED — supports post-plan C5
contribution]. The Zenodo record turns out to be much larger than the
prompt's "~4.7k images" estimate: 19 per-crop zip files totalling
~14 GB.

Phase 4 deferred full OLID I download. C5 evaluation in Phase 11 will
set ``OLID_FULL_DOWNLOAD = True`` and re-run this script.

Defaults:
- ``OLID_FULL_DOWNLOAD = False`` → downloads a single crop archive
  (smoke sample) plus the dataset's class-distribution spreadsheet. The
  smoke sample is enough to validate the multi-label dataset class
  pipeline; it is **not** sufficient for the C5 evaluation in Phase 11.
- ``OLID_FULL_DOWNLOAD = True`` → enumerates the Zenodo record's files
  via the public API and downloads every archive. Allow ~30–60 minutes
  and ~14 GB of disk.

Smoke-sample policy: tomato is the most label-diverse crop, but
``tomato__part_1.zip`` is 1.39 GB which exceeds the 500 MB threshold
the supervisor set for the sample. Fall back to
``bottle_gourd__part_1.zip`` (~186 MB) instead.

Target: ``data/causation/olid_i/raw/``. Note: ``data/causation/`` is
[ADDED] — not in master reference §41 because C5 was post-plan.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    http_download_with_retry,
    is_directory_populated,
    sha256sum,
    unzip_into,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

ZENODO_RECORD = "8105154"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Toggle to True in Phase 11 to fetch the full ~14 GB dataset.
OLID_FULL_DOWNLOAD: bool = False

# Smoke-sample components: a metadata spreadsheet + one crop archive.
SMOKE_SAMPLE_FILES: tuple[str, ...] = (
    "class_distribution.xlsx",
    "bottle_gourd__part_1.zip",
)

CAUSATION_DIR = DATA_DIR / "causation"
RAW_DIR: Path = CAUSATION_DIR / "olid_i" / "raw"


def _zenodo_file_listing() -> list[dict[str, object]]:
    """Return the Zenodo record's file listing (key, size, content URL)."""
    import requests  # noqa: PLC0415

    resp = requests.get(ZENODO_API_URL, timeout=60)
    resp.raise_for_status()
    return resp.json().get("files", [])


def _download_one(filename: str, file_url: str, target: Path) -> Path:
    target_path = target / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        _LOGGER.info("%s already present, skipping.", filename)
        return target_path
    http_download_with_retry(file_url, target_path)
    _LOGGER.info("%s SHA-256: %s", filename, sha256sum(target_path))
    return target_path


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("OLID I already present at %s — skipping.", RAW_DIR)
        return 0

    listing = _zenodo_file_listing()
    by_key = {entry["key"]: entry for entry in listing}

    if OLID_FULL_DOWNLOAD:
        wanted = list(by_key.keys())
        _LOGGER.info(
            "OLID_FULL_DOWNLOAD=True — fetching all %d files (~14 GB).",
            len(wanted),
        )
    else:
        # Verify the smoke-sample files actually exist in the record.
        missing = [name for name in SMOKE_SAMPLE_FILES if name not in by_key]
        if missing:
            _LOGGER.error(
                "Smoke-sample files missing from Zenodo record: %s. "
                "Refusing to download a wrong subset.",
                missing,
            )
            return 1
        wanted = list(SMOKE_SAMPLE_FILES)
        _LOGGER.info(
            "OLID_FULL_DOWNLOAD=False — smoke sample only (%d files). "
            "Set OLID_FULL_DOWNLOAD=True in Phase 11 to pull the full "
            "dataset.",
            len(wanted),
        )

    downloaded_zips: list[Path] = []
    for name in wanted:
        entry = by_key[name]
        file_url = entry["links"]["self"]  # type: ignore[index]
        path = _download_one(name, str(file_url), RAW_DIR)
        if path.suffix.lower() == ".zip":
            downloaded_zips.append(path)

    for zip_path in downloaded_zips:
        unzip_into(zip_path, RAW_DIR)
        # Keep the original archive — useful for re-verification.

    _LOGGER.info("OLID I (%s) ready at %s",
                 "full" if OLID_FULL_DOWNLOAD else "smoke sample", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
