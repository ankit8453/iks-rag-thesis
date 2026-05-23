"""Download the Sirajganj Soil Moisture dataset [ADDED].

Source: Mendeley Data DOI ``10.17632/skcc44yvvg.2`` (version 2,
published 26 August 2025).

Citation
--------
Raihan A., Fayaz S.M., Ahmed J., Hossain M. (2025). "Soil Moisture
Dataset for Image Based Soil Classification." Mendeley Data, V2,
doi:10.17632/skcc44yvvg.2

Content
-------
Smartphone images (Sony Xperia 1 Mark II) captured in Sirajganj,
Bangladesh under natural outdoor lighting. Three labelled classes:
Wet, Moderate, Dry. The Phase-4-fix prompt estimated ~1,177 images at
~500 MB, but the actual v2 archive is ~4.49 GB (the upstream dataset
appears to have grown between v1 and v2). The image count is verified
post-extraction.

Role
----
Supervises the soil module's ``moisture_appearance`` head (master plan
§14: ``Wet → wet``, ``Moderate → moist``, ``Dry → dry``). This dataset
is the post-Phase-4 reconciliation addition; it was not in the
original Phase 4 prompt because it was identified after that prompt
was locked.

Target: ``data/soil/sirajganj_moisture/raw/``.

Idempotent: skip if ``raw/`` already contains the extracted dataset.

Download strategy
-----------------
Queries Mendeley's public-API endpoint
``https://data.mendeley.com/public-api/datasets/{accession}/files?
folder_id=root&version={version}`` to discover the current signed
download URL (cached for 100+ years per the response's
``download_expiry_time``). Then fetches via ``http_download_with_retry``.

# FALLBACK (manual signed URL, as observed on 2026-05-23):
#   https://data.mendeley.com/public-files/datasets/skcc44yvvg/files/
#   2476b34c-0b31-40c6-b2c3-c98dfc45ec8f/file_downloaded
#   SHA-256: bf831c138cecda3ed92df3d36415b896c0f48dc3984a320fa04c20692e7b9e4b
# If Mendeley breaks the public-API endpoint, hardcode the URL above.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    http_download_with_retry,
    is_directory_populated,
    sha256sum,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_SOIL_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

MENDELEY_ACCESSION = "skcc44yvvg"
MENDELEY_VERSION = "2"
MENDELEY_FILES_API = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ACCESSION}/"
    f"files?folder_id=root&version={MENDELEY_VERSION}"
)
EXPECTED_SHA256 = "bf831c138cecda3ed92df3d36415b896c0f48dc3984a320fa04c20692e7b9e4b"

RAW_DIR: Path = DATA_SOIL_DIR / "sirajganj_moisture" / "raw"


def _resolve_download_url() -> tuple[str, str]:
    """Query Mendeley's public API for the signed download URL.

    Returns ``(filename, download_url)``.
    """
    import requests  # noqa: PLC0415

    resp = requests.get(MENDELEY_FILES_API, timeout=30, headers={"Accept": "application/json"})
    resp.raise_for_status()
    entries = resp.json()
    if not entries:
        raise RuntimeError(f"Mendeley API returned no files for {MENDELEY_ACCESSION}.")
    entry = entries[0]
    filename = entry["filename"]
    url = entry["content_details"]["download_url"]
    return filename, url


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("Sirajganj moisture already present at %s — skipping.", RAW_DIR)
        return 0

    filename, url = _resolve_download_url()
    _LOGGER.info("Mendeley download URL resolved: %s", filename)

    archive_path = RAW_DIR / filename
    http_download_with_retry(url, archive_path)

    actual_hash = sha256sum(archive_path)
    _LOGGER.info("%s SHA-256: %s", filename, actual_hash)
    if actual_hash != EXPECTED_SHA256:
        _LOGGER.warning(
            "SHA-256 mismatch! Expected %s, got %s. The Mendeley dataset may "
            "have been re-published. Update EXPECTED_SHA256 in this script "
            "after verifying the new archive is the same content.",
            EXPECTED_SHA256,
            actual_hash,
        )

    _LOGGER.info("Unzipping %s -> %s", archive_path, RAW_DIR)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(RAW_DIR)

    # Free disk: drop the archive after successful extraction.
    try:
        archive_path.unlink()
    except OSError:
        pass

    _LOGGER.info("Sirajganj moisture dataset extracted to %s", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
