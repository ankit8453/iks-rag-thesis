"""Batch-segment an HF disease dataset + cache the masks (Phase 5-R Part 2).

The Phase 5-R cascade pulls images from HuggingFace datasets (same
pattern as the original Phase 5 trainer, so it works on a fresh Colab
runtime). This module walks an HF split's rows in order, runs the
Part-1 :func:`src.disease.segment.segment` router on each image, and
saves the resulting mask under :data:`MASK_CACHE_ROOT` keyed by
``(dataset_id, split, row_idx)``. The trainer then loads
``mask_path_for(dataset_id, split, idx).png`` for the row it's about
to randomize.

Why HF-row indexing (and not a filename hash):

- The original Phase 5 trainer's loaders are also indexed by HF row,
  so the train / val / test ordering already matches the trainer's
  worldview. No filename / path translation needed.
- ``load_dataset(repo)`` is deterministic at fixed dataset revisions.
- Resume-friendly: the cache survives a Colab session timeout and a
  re-run hits zero new segmentations.

Storage layout::

    data/plant_disease/_masks/
    ├── plantvillage/
    │   ├── train/000000.png  000001.png  ...
    │   ├── val/000000.png    ...
    │   └── _segmentation_log.json
    └── plantdoc/
        └── ...

Hard rules from the prompt:

- Idempotent: re-running skips cached masks.
- Log % flagged per dataset+split.
- Flagged masks (foreground < 5% or > 95%) are still written so the
  randomized dataset can detect + fall back to raw at training time —
  the trainer treats "flagged" rows as no-randomize.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.disease.segment import SegmentStyle, segment
from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    import numpy as np  # noqa: F401

_LOGGER = get_logger(__name__)

# Where masks land. Gitignored.
MASK_CACHE_ROOT: Path = PROJECT_ROOT / "data" / "plant_disease" / "_masks"

# Per-split log filename. One JSON per dataset records the flagged list
# across all splits the trainer cached.
_LOG_FILENAME: str = "_segmentation_log.json"


@dataclass
class CacheStats:
    """Per-(dataset, split) cache build summary."""

    dataset: str
    split: str
    style: str
    total: int
    cached_already: int
    newly_segmented: int
    flagged: int
    failures: int
    flagged_fraction: float
    elapsed_seconds: float


# --------------------------------------------------------------------- #
# Path helpers (HF row indexed)
# --------------------------------------------------------------------- #


def mask_path_for(dataset: str, split: str, row_idx: int) -> Path:
    """Where the mask PNG for HF row ``(dataset, split, row_idx)`` lives."""
    return MASK_CACHE_ROOT / dataset / split / f"{int(row_idx):06d}.png"


def dataset_log_path(dataset: str) -> Path:
    return MASK_CACHE_ROOT / dataset / _LOG_FILENAME


def _load_existing_log(dataset: str) -> dict[str, Any]:
    p = dataset_log_path(dataset)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "dataset": dataset,
        "splits": {},
        "flagged_keys": [],
    }


def _save_log(dataset: str, log: dict[str, Any]) -> None:
    p = dataset_log_path(dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(log, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- #
# Single image
# --------------------------------------------------------------------- #


def segment_and_cache_one_image(
    pil_image: Any,
    out_path: Path,
    style: SegmentStyle,
) -> tuple[bool, float]:
    """Segment one PIL image and write the mask PNG to ``out_path``.

    Returns ``(flagged, foreground_fraction)``.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    result = segment(pil_image, style=style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(result.mask, mode="L").save(out_path)
    return result.flagged_as_failure, result.foreground_fraction


# --------------------------------------------------------------------- #
# Dataset+split-level driver — works directly on an HF split
# --------------------------------------------------------------------- #


def build_mask_cache_from_hf(
    dataset_repo: str,
    dataset_id: str,
    split: str,
    style: SegmentStyle,
    *,
    log_every: int = 100,
) -> CacheStats:
    """Walk one HF split's rows and segment+cache every uncached row.

    Parameters
    ----------
    dataset_repo
        HF dataset repo id (``ankit-iiitdmj/iks-plantvillage`` etc.).
    dataset_id
        Short identifier used as the local cache subdirectory name
        (``"plantvillage"`` / ``"plantdoc"``). Must match the
        ``dataset_id`` the trainer will pass to the randomized dataset.
    split
        ``"train"`` / ``"val"`` / ``"test"``.
    style
        ``"lab"`` (classical) for PlantVillage; ``"field"`` (rembg) for
        PlantDoc.
    """
    from datasets import load_dataset  # noqa: PLC0415

    _LOGGER.info(
        "Building mask cache for %s split=%s style=%s ...",
        dataset_id, split, style,
    )
    hf = load_dataset(dataset_repo, split=split)
    total = len(hf)

    cached_already = 0
    newly_segmented = 0
    flagged_count = 0
    failures = 0
    flagged_keys_this_split: list[str] = []

    t0 = time.monotonic()
    for idx in range(total):
        mask_out = mask_path_for(dataset_id, split, idx)
        if mask_out.is_file() and mask_out.stat().st_size > 0:
            cached_already += 1
            if (idx + 1) % log_every == 0:
                _LOGGER.info(
                    "[%s/%s] %d/%d done — %d cached, %d new, %d flagged",
                    dataset_id, split, idx + 1, total,
                    cached_already, newly_segmented, flagged_count,
                )
            continue
        try:
            row = hf[idx]
            pil = row["image"].convert("RGB")
            flagged, _fg = segment_and_cache_one_image(pil, mask_out, style=style)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "[%s/%s] row %d: segmentation FAILED: %s — will fall back to raw at train time.",
                dataset_id, split, idx, exc,
            )
            failures += 1
            continue
        newly_segmented += 1
        if flagged:
            flagged_count += 1
            flagged_keys_this_split.append(f"{split}/{idx:06d}")
        if (idx + 1) % log_every == 0:
            _LOGGER.info(
                "[%s/%s] %d/%d done — %d cached, %d new, %d flagged",
                dataset_id, split, idx + 1, total,
                cached_already, newly_segmented, flagged_count,
            )

    elapsed = time.monotonic() - t0
    denom = max(1, newly_segmented)
    stats = CacheStats(
        dataset=dataset_id, split=split, style=style,
        total=total, cached_already=cached_already,
        newly_segmented=newly_segmented, flagged=flagged_count,
        failures=failures,
        flagged_fraction=flagged_count / denom,
        elapsed_seconds=float(elapsed),
    )

    # Merge into the persistent per-dataset log so a later split's run
    # doesn't overwrite an earlier one's flagged list.
    log = _load_existing_log(dataset_id)
    log["splits"][split] = {
        "style": style, "total": total,
        "cached_already": cached_already,
        "newly_segmented": newly_segmented,
        "flagged": flagged_count,
        "failures": failures,
        "flagged_fraction": stats.flagged_fraction,
        "elapsed_seconds": stats.elapsed_seconds,
    }
    # ``flagged_keys`` is the union across splits — kept as a list so
    # the JSON file is readable; converted to a set in ``load_flagged_set``.
    union = set(log.get("flagged_keys") or [])
    union.update(flagged_keys_this_split)
    log["flagged_keys"] = sorted(union)
    _save_log(dataset_id, log)

    _LOGGER.info(
        "[%s/%s] cache built: total=%d, new=%d, flagged=%d (%.1f%%), failures=%d, elapsed=%.0fs",
        dataset_id, split, total, newly_segmented, flagged_count,
        100 * stats.flagged_fraction, failures, elapsed,
    )
    return stats


# --------------------------------------------------------------------- #
# Trainer-time helpers
# --------------------------------------------------------------------- #


def load_flagged_set(dataset: str) -> set[str]:
    """Return the set of ``"<split>/<row_idx:06d>"`` keys whose mask was
    flagged at cache time.

    The randomized trainer uses this to decide: composite (use the
    cached mask) vs fall back to the raw HF image.
    """
    log_path = dataset_log_path(dataset)
    if not log_path.is_file():
        return set()
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return set(data.get("flagged_keys") or [])


def is_flagged(dataset: str, split: str, row_idx: int) -> bool:
    key = f"{split}/{int(row_idx):06d}"
    return key in load_flagged_set(dataset)


__all__ = [
    "CacheStats",
    "MASK_CACHE_ROOT",
    "build_mask_cache_from_hf",
    "dataset_log_path",
    "is_flagged",
    "load_flagged_set",
    "mask_path_for",
    "segment_and_cache_one_image",
]
