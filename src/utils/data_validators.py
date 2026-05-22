"""Image-dataset validation utilities (Phase 4 §B).

Per supervisor guardrail #6: image integrity must be validated before
splitting; corrupt / zero-byte / truncated files are flagged and
excluded from splits but **never deleted** — the exclusion list is
saved so a human can audit.

The validator opens each file with PIL twice (once to ``open``, once to
``verify`` after a fresh open) because ``Image.verify`` is destructive
to the file handle and only catches a subset of truncation cases.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

_MIN_DIM = 64
_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


class ImageValidationReport(BaseModel):
    """Outcome of validating one dataset's image directory.

    Attributes
    ----------
    dataset_name : str
        Friendly identifier, e.g. ``"plantvillage"``.
    total_files : int
        Total image-extension files scanned under the root.
    valid_files : int
        Files that passed every check.
    corrupt_files : list[str]
        Paths of files that failed at least one check, expressed
        relative to the dataset root.
    failure_reasons : dict[str, str]
        ``relative_path -> short reason`` for each entry in
        ``corrupt_files``.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    total_files: int
    valid_files: int
    corrupt_files: list[str]
    failure_reasons: dict[str, str]


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _check_image(path: Path) -> tuple[bool, str]:
    """Return ``(is_valid, reason)`` for a single file.

    PIL is imported lazily so this module remains importable in
    pre-install environments.
    """
    if not path.exists():
        return False, "missing"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat-failed: {exc}"
    if size == 0:
        return False, "zero-byte"

    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    # Pass 1: verify() — catches truncated headers but invalidates the
    # file pointer afterwards.
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"verify-failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"verify-error: {exc}"

    # Pass 2: actually decode to confirm dimensions + mode.
    try:
        with Image.open(path) as img:
            img.load()
            mode = img.mode
            width, height = img.size
    except Exception as exc:  # noqa: BLE001
        return False, f"decode-failed: {exc}"

    if width < _MIN_DIM or height < _MIN_DIM:
        return False, f"too-small: {width}x{height}"

    # Accept RGB, RGBA, L (will be promoted to RGB by the dataset
    # transforms), and palette modes.
    if mode not in {"RGB", "RGBA", "L", "P", "CMYK"}:
        return False, f"unexpected-mode: {mode}"

    return True, ""


def validate_image_directory(root: Path, dataset_name: str) -> ImageValidationReport:
    """Walk ``root`` and validate every image-extension file under it.

    Parameters
    ----------
    root : Path
        Directory to validate (recursively).
    dataset_name : str
        Friendly identifier carried through into the report.

    Returns
    -------
    ImageValidationReport
        ``corrupt_files`` contains paths relative to ``root``.

    Notes
    -----
    Does not delete or modify any file. The exclusion list returned here
    is consumed by :mod:`src.utils.data_splits` to drop bad paths from
    every generated split.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Validation root does not exist: {root}")

    total = 0
    valid = 0
    corrupt: list[str] = []
    reasons: dict[str, str] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not _is_image_path(path):
            continue
        total += 1
        ok, reason = _check_image(path)
        rel = path.relative_to(root).as_posix()
        if ok:
            valid += 1
        else:
            corrupt.append(rel)
            reasons[rel] = reason

    if total > 0 and len(corrupt) / total > 0.01:
        _LOGGER.warning(
            "Dataset %s: %d/%d (%.2f%%) corrupt files — exceeds 1%% threshold.",
            dataset_name,
            len(corrupt),
            total,
            100 * len(corrupt) / total,
        )
    else:
        _LOGGER.info(
            "Dataset %s: %d/%d files valid (%d corrupt).",
            dataset_name,
            valid,
            total,
            len(corrupt),
        )

    return ImageValidationReport(
        dataset_name=dataset_name,
        total_files=total,
        valid_files=valid,
        corrupt_files=corrupt,
        failure_reasons=reasons,
    )


def write_corrupt_list(report: ImageValidationReport, results_dir: Path) -> Path | None:
    """Persist a corrupt-file list to ``results_dir`` if any were found.

    Returns the path written, or None if there was nothing to write.
    """
    if not report.corrupt_files:
        return None
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"corrupt_files_{report.dataset_name}.txt"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Corrupt files in dataset '{report.dataset_name}'\n")
        fh.write(f"# Total scanned: {report.total_files}, valid: {report.valid_files}\n")
        for rel in report.corrupt_files:
            reason = report.failure_reasons.get(rel, "?")
            fh.write(f"{rel}\t{reason}\n")
    _LOGGER.info("Wrote corrupt list -> %s", out_path)
    return out_path


__all__ = [
    "ImageValidationReport",
    "validate_image_directory",
    "write_corrupt_list",
]
