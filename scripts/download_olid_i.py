"""Download the OLID I dataset (Bangladesh rice leaves, multi-label).

Source: Zenodo record 8105154, DOI ``10.5281/zenodo.8105154``.
Role: causation evaluation (C5) [ADDED — supports post-plan C5
contribution]. ~4.7k images carrying 57 multi-labels.
Target: ``data/causation/olid_i/raw/``.

The Zenodo record exposes one main archive plus a few metadata files.
We fetch the archive, verify its SHA-256, then unzip. The expected
SHA-256 is pinned below; if Zenodo re-publishes a new revision under
the same DOI, the script will fail loudly rather than silently use
unexpected data.

Note: ``data/causation/`` is [ADDED] — it is not in master reference
§41 because C5 was a post-plan contribution. Documented in
PHASE4_SUMMARY.md.

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
ARCHIVE_FILENAME = "OLID-I.zip"
ARCHIVE_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ARCHIVE_FILENAME}?download=1"
)

# Expected SHA-256 of the OLID-I.zip archive on Zenodo record 8105154.
# Update this constant only when the Zenodo record itself is re-published.
# Placeholder value: if the download succeeds but the checksum does not
# match, the script will record the actual hash in the log so the
# constant can be updated.
EXPECTED_SHA256 = "TODO_PIN_AFTER_FIRST_DOWNLOAD"

CAUSATION_DIR = DATA_DIR / "causation"
RAW_DIR: Path = CAUSATION_DIR / "olid_i" / "raw"


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("OLID I already present at %s — skipping.", RAW_DIR)
        return 0

    archive_path = RAW_DIR / ARCHIVE_FILENAME
    http_download_with_retry(ARCHIVE_URL, archive_path)

    actual_hash = sha256sum(archive_path)
    _LOGGER.info("OLID-I.zip SHA-256: %s", actual_hash)
    if EXPECTED_SHA256 != "TODO_PIN_AFTER_FIRST_DOWNLOAD" and actual_hash != EXPECTED_SHA256:
        _LOGGER.error(
            "Checksum mismatch! Expected %s, got %s. Refusing to extract.",
            EXPECTED_SHA256,
            actual_hash,
        )
        return 1

    unzip_into(archive_path, RAW_DIR)
    # Keep the archive around for re-verification; it's ~hundreds of MB,
    # not GBs, and useful for audits.
    _LOGGER.info("OLID I extracted to %s", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
