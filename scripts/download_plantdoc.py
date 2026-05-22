"""Download the PlantDoc dataset (cropped variant).

Source: GitHub ``pratikkayal/PlantDoc-Dataset`` (downloaded as a zip
archive rather than git-cloned).
Role: disease real-field evaluation (PDF §20). ~2.5k images, 27 classes.
Target: ``data/plant_disease/plantdoc/raw/``.

Why zip instead of git clone: a handful of files in the upstream repo
have URL-query-string suffixes (e.g.
``test/Bell_pepper leaf/IMG_1629.JPG?1507122477.jpg``). The ``?`` is
illegal in Windows filenames so ``git checkout`` aborts with "invalid
path". Downloading a zip and extracting it skips the offending files
silently rather than failing the whole operation. The cropped variant
(which is what we use) lives under a ``cropped`` subdirectory inside
the extracted archive; the other variants are discarded.

Idempotent.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import (  # noqa: E402
    ensure_raw_dir,
    http_download_with_retry,
    is_directory_populated,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

# Codeload zips of master / main; tried in order.
ZIP_URLS = [
    "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master",
    "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/main",
]
RAW_DIR: Path = DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw"


def _safe_extractall(zip_path: Path, target: Path) -> int:
    """Extract members one-by-one, skipping any that fail on Windows.

    Returns the number of members successfully extracted.
    """
    ok = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            try:
                zf.extract(member, target)
                ok += 1
            except (OSError, ValueError) as exc:
                _LOGGER.warning("Skipping unextractable member %r: %s", member.filename, exc)
    return ok


def main() -> int:
    ensure_raw_dir(RAW_DIR)
    if is_directory_populated(RAW_DIR):
        _LOGGER.info("PlantDoc already present at %s — skipping.", RAW_DIR)
        return 0

    parent = RAW_DIR.parent
    zip_path = parent / "_plantdoc_repo.zip"
    extract_dir = parent / "_plantdoc_repo"

    last_exc: Exception | None = None
    for url in ZIP_URLS:
        try:
            http_download_with_retry(url, zip_path)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _LOGGER.warning("Zip URL %s failed: %s", url, exc)
            continue
    else:
        _LOGGER.error("All PlantDoc zip URLs failed; last error: %s", last_exc)
        return 1

    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    ok = _safe_extractall(zip_path, extract_dir)
    _LOGGER.info("Extracted %d members from PlantDoc zip.", ok)

    # PlantDoc-Dataset upstream layout (as of 2026-05): the repo root
    # contains ``train/<class>/*.jpg`` and ``test/<class>/*.jpg``. The
    # original-paper "cropped variant" refers to the fact that all
    # images are cropped leaf shots; there is no separate ``cropped``
    # subdirectory. We merge train + test into a single class-folder
    # layout under RAW_DIR; the stratified-split step will produce its
    # own train/val/test JSON later.
    repo_root_candidates = [c for c in extract_dir.iterdir() if c.is_dir()]
    if not repo_root_candidates:
        _LOGGER.error("Extraction directory %s appears empty.", extract_dir)
        return 1
    repo_root = repo_root_candidates[0]

    source_dirs = [repo_root / "train", repo_root / "test"]
    available_sources = [d for d in source_dirs if d.is_dir()]
    if not available_sources:
        _LOGGER.error(
            "Neither train/ nor test/ found under %s; PlantDoc layout has "
            "changed. Leaving extraction in place for manual inspection.",
            repo_root,
        )
        return 1

    for source in available_sources:
        for class_dir in source.iterdir():
            if not class_dir.is_dir():
                continue
            target = RAW_DIR / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for image in class_dir.iterdir():
                if not image.is_file():
                    continue
                dest = target / image.name
                if dest.exists():
                    # Different source folders can produce a name clash;
                    # keep both with a suffix.
                    dest = target / f"{image.stem}__{source.name}{image.suffix}"
                shutil.move(str(image), str(dest))

    # Clean up.
    shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        zip_path.unlink()
    except OSError:
        pass

    _LOGGER.info("PlantDoc cropped variant ready at %s", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
