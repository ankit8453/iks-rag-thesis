"""Integrate the VIT Vellore ``latha-soil`` texture dataset into ``data/soil/vit_texture/``.

Source repository: ``https://github.com/phd-latha/latha-soil`` (Reddy &
Gopinath, Nature Sci. Rep. 2025, doi:10.1038/s41598-025-17384-5).

This is a supplementary texture-axis dataset that sits alongside IRSID
under ``data/soil/``. The two datasets are kept on disk as separate
directories; Phase 6 training code will combine them via PyTorch
``ConcatDataset`` rather than by filesystem merging.

The script is idempotent:

- If ``SCRATCH_DIR`` already exists with a populated checkout (.git clone
  OR an extracted zip), it is reused.
- If absent, the script attempts ``git clone`` from ``REPO_URL``. The
  clone may fail in sandboxed environments — in that case the user is
  expected to manually drop the GitHub zip at ``REPO_URL`` and re-run
  after extracting to ``SCRATCH_DIR``.

Filename canonicalisation matches the spec in PHASE5_VIT_INTEGRATION_PROMPT.md:
trailing digits stripped, lowercase-snake-case, fuzzy match against the
authoritative ``classes.txt``. Bare-prefix filenames (e.g. ``clay 1.png``)
that do not auto-match via the documented fuzzy rules fall through to an
explicit ``BARE_NAME_ALIASES`` table; the alias choices were confirmed
with Ankit before writing this script.

Outputs:

- ``data/soil/vit_texture/raw/<class>/<seq>.<ext>`` — class-bucketed images
- ``data/soil/vit_texture/_review/`` — any file whose class is ambiguous
- ``data/soil/vit_texture/INTEGRATION_AUDIT.json`` — provenance manifest
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, UnidentifiedImageError  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_SOIL_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

REPO_URL = "https://github.com/phd-latha/latha-soil"
SCRATCH_DIR = Path("/tmp/latha-soil")

VIT_ROOT: Path = DATA_SOIL_DIR / "vit_texture"
TARGET_DIR: Path = VIT_ROOT / "raw"
REVIEW_DIR: Path = VIT_ROOT / "_review"
METADATA_PATH: Path = VIT_ROOT / "INTEGRATION_AUDIT.json"

# Fallback class list — used only if ``classes.txt`` is missing or empty.
# Conservative names per the prompt's "Expected canonical class names".
_FALLBACK_CLASSES: list[str] = [
    "clay",
    "sandy_clay",
    "sandy_loam",
    "loamy_sand",
    "loam",
    "silt",
    "sandy_soil",
]

# Bare-prefix filename aliases (e.g. ``clay 1.png``). The documented
# fuzzy rules in ``canonicalize_class`` handle ``sandy`` / ``loamy`` /
# ``silt`` automatically via ``+ "_soil"``; only ``clay`` needs an
# explicit alias because the authoritative class is ``clayey_soils``
# (plural), not ``clay_soil``.
BARE_NAME_ALIASES: dict[str, str] = {
    "clay": "clayey_soils",
}

_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass
class _IntegrationStats:
    classes_used: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    review: list[str] = field(default_factory=list)
    pil_failed: list[str] = field(default_factory=list)
    total_seen: int = 0


def _canonicalize_string(text: str) -> str:
    """Lowercase + collapse whitespace + spaces-to-underscores."""
    return re.sub(r"\s+", " ", text).strip().lower().replace(" ", "_")


def canonicalize_class(filename: str, known_classes: list[str]) -> str | None:
    """Map a raw filename to a canonical class name, or ``None`` if ambiguous.

    Steps:
      1. Strip extension.
      2. Strip trailing whitespace + digits + whitespace
         (e.g. ``"Sandy Clay 11"`` -> ``"Sandy Clay"``).
      3. Canonicalise to lowercase-snake-case.
      4. Direct lookup in ``known_classes``.
      5. Fuzzy: try dropping a trailing ``_soil`` suffix, then try appending
         it (covers ``Silt`` vs ``Silt soil`` etc.).
      6. ``BARE_NAME_ALIASES`` fallback (currently only ``clay``).
      7. Otherwise return ``None``.
    """
    stem = Path(filename).stem
    base = re.sub(r"\s*\d+\s*$", "", stem).strip()
    canonical = _canonicalize_string(base)

    if canonical in known_classes:
        return canonical

    if canonical.endswith("_soil"):
        alt = canonical[: -len("_soil")]
        if alt in known_classes:
            return alt
    alt = canonical + "_soil"
    if alt in known_classes:
        return alt

    if canonical in BARE_NAME_ALIASES:
        candidate = BARE_NAME_ALIASES[canonical]
        if candidate in known_classes:
            return candidate

    return None


def _read_known_classes(scratch_dir: Path) -> list[str]:
    classes_file = scratch_dir / "classes.txt"
    if not classes_file.is_file():
        _LOGGER.warning("classes.txt missing — using fallback class list.")
        return list(_FALLBACK_CLASSES)
    raw = classes_file.read_text(encoding="utf-8").strip()
    if not raw:
        _LOGGER.warning("classes.txt is empty — using fallback class list.")
        return list(_FALLBACK_CLASSES)
    lines = [line for line in raw.splitlines() if line.strip()]
    canonical_list = [_canonicalize_string(line) for line in lines]
    _LOGGER.info("classes.txt: %d entries → %s", len(canonical_list), canonical_list)
    return canonical_list


def _ensure_scratch(scratch_dir: Path) -> None:
    """Clone the repo to ``scratch_dir`` if it isn't already populated."""
    if scratch_dir.is_dir() and any(scratch_dir.iterdir()):
        _LOGGER.info("Re-using existing scratch checkout at %s", scratch_dir)
        return
    _LOGGER.info("Cloning %s -> %s", REPO_URL, scratch_dir)
    scratch_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(scratch_dir)],
        check=True,
    )


def _resolve_commit_sha(scratch_dir: Path) -> str | None:
    """Return short commit SHA if scratch dir is a git checkout, else ``None``."""
    if not (scratch_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(scratch_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _LOGGER.warning("Could not resolve commit SHA: %s", exc)
        return None


def _validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return False
    return True


def _reset_target_dirs() -> None:
    """Wipe ``raw/`` and ``_review/`` so the integration is idempotent.

    The script re-derives all images from the scratch dir on every run; if
    we appended into existing class folders instead, re-runs would duplicate
    every file with fresh sequence indices.
    """
    for d in (TARGET_DIR, REVIEW_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _integrate(scratch_dir: Path, known_classes: list[str]) -> _IntegrationStats:
    stats = _IntegrationStats(classes_used=list(known_classes))
    _reset_target_dirs()

    sources = sorted(
        p for p in scratch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    _LOGGER.info("Found %d candidate image files in scratch dir.", len(sources))

    for src in sources:
        stats.total_seen += 1
        if not _validate_image(src):
            _LOGGER.warning("PIL rejected %s — skipping.", src.name)
            stats.pil_failed.append(src.name)
            continue

        cls = canonicalize_class(src.name, known_classes)
        if cls is None:
            dest = REVIEW_DIR / src.name
            shutil.copy2(src, dest)
            stats.review.append(src.name)
            _LOGGER.warning("Unrecognised class for %s — routed to _review/.", src.name)
            continue

        class_dir = TARGET_DIR / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        stats.class_counts[cls] = stats.class_counts.get(cls, 0) + 1
        seq = stats.class_counts[cls]
        ext = src.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        dest = class_dir / f"{seq:04d}{ext}"
        shutil.copy2(src, dest)

    return stats


def _write_audit(
    stats: _IntegrationStats,
    *,
    commit_sha: str | None,
    classes_file_present: bool,
) -> None:
    payload = {
        "source_url": REPO_URL,
        "source_commit_sha": commit_sha,
        "source_note": (
            "Repo was supplied as a GitHub zip archive (latha-soil-main.zip) "
            "and extracted into the scratch directory; no .git history was "
            "available, so source_commit_sha is null."
            if commit_sha is None else
            "Cloned from GitHub at integration time."
        ),
        "integration_date": date.today().isoformat(),
        "classes_file_present": classes_file_present,
        "canonical_classes": stats.classes_used,
        "class_counts": dict(sorted(stats.class_counts.items())),
        "files_total_seen": stats.total_seen,
        "files_placed": sum(stats.class_counts.values()),
        "files_to_review": stats.review,
        "files_pil_failed": stats.pil_failed,
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _LOGGER.info("Wrote %s", METADATA_PATH)


def _print_summary(stats: _IntegrationStats) -> None:
    placed = sum(stats.class_counts.values())
    print()
    print("VIT texture dataset integrated:")
    print(f"  Total in raw/: {placed} images across {len(stats.class_counts)} classes")
    print("  Class breakdown:")
    for cls, n in sorted(stats.class_counts.items()):
        print(f"    {cls}: {n}")
    print(f"  To review (manual): {len(stats.review)} images")
    print(f"  Skipped (corrupt): {len(stats.pil_failed)} images")


def main() -> int:
    _ensure_scratch(SCRATCH_DIR)
    classes_file_present = (SCRATCH_DIR / "classes.txt").is_file()
    known_classes = _read_known_classes(SCRATCH_DIR)

    stats = _integrate(SCRATCH_DIR, known_classes)
    commit_sha = _resolve_commit_sha(SCRATCH_DIR)
    _write_audit(stats, commit_sha=commit_sha, classes_file_present=classes_file_present)
    _print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
