"""Batch-segment a dataset's images + cache the masks (Phase 5-R Part 2).

Phase 5-R's on-the-fly random-background compositor needs a leaf mask
per training image. Re-running :func:`src.disease.segment.segment` every
batch is feasible but burns the U2Net forward pass on the CPU GPU,
multiplied by epochs. Cheaper to segment ONCE up-front and cache the
boolean mask as a single-channel PNG.

Storage layout (mirrors the dataset's relative path so cache lookup is
``relpath -> mask path`` without a separate manifest):

::

    data/plant_disease/_masks/
    ├── plantvillage/
    │   ├── Apple___Apple_scab/<image>.png
    │   ├── …
    │   └── _segmentation_log.json   (run summary + flagged list)
    └── plantdoc/
        ├── Apple Scab Leaf/<image>.png
        ├── …
        └── _segmentation_log.json

Hard rules from the prompt:

- Idempotent: re-running skips cached masks (re-hashing every image is
  not needed; we trust the path-based cache).
- Log % flagged per dataset.
- Flagged masks (foreground < 5% or > 95%) are still written to disk so
  the random-bg compositor can detect + fall back to the raw image at
  training time — the trainer treats "flagged" rows as no-randomize.
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
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)

# Where masks land. Sibling to ``raw/`` under each disease dataset so a
# gitignore rule (``data/plant_disease/_masks/``) hides the cache.
MASK_CACHE_ROOT: Path = PROJECT_ROOT / "data" / "plant_disease" / "_masks"

# Filename for the per-dataset run summary.
_LOG_FILENAME: str = "_segmentation_log.json"


@dataclass
class CacheStats:
    """Per-dataset cache build summary."""

    dataset: str
    style: str
    total: int
    cached_already: int
    newly_segmented: int
    flagged: int
    failures: int
    flagged_fraction: float
    elapsed_seconds: float


# --------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------- #


def mask_path_for(dataset: str, rel_image_path: str | Path) -> Path:
    """Return where the mask PNG for ``rel_image_path`` should live."""
    p = Path(rel_image_path)
    return MASK_CACHE_ROOT / dataset / p.with_suffix(".png")


def dataset_log_path(dataset: str) -> Path:
    return MASK_CACHE_ROOT / dataset / _LOG_FILENAME


# --------------------------------------------------------------------- #
# Single image
# --------------------------------------------------------------------- #


def segment_and_cache_one(
    image_path: Path | str,
    out_path: Path,
    style: SegmentStyle,
) -> tuple[bool, float]:
    """Segment one image and write the mask to ``out_path``.

    Returns ``(flagged, foreground_fraction)``. Caller uses these to
    decide whether the row falls back to raw at training time.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    result = segment(image_path, style=style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(result.mask, mode="L").save(out_path)
    return result.flagged_as_failure, result.foreground_fraction


# --------------------------------------------------------------------- #
# Dataset-level driver
# --------------------------------------------------------------------- #


def build_mask_cache(
    dataset: str,
    style: SegmentStyle,
    image_iter: list[tuple[str, Path]],
    *,
    log_every: int = 100,
) -> CacheStats:
    """Walk ``image_iter`` and segment every image whose mask is not yet
    cached. Idempotent.

    Parameters
    ----------
    dataset
        Short identifier — used as the cache subdirectory name. Pass
        ``"plantvillage"`` / ``"plantdoc"`` (Paddy Doctor is NOT
        randomized per the Part-1 verdict, so its mask cache is
        deliberately not built here).
    style
        ``"lab"`` (classical) for PlantVillage; ``"field"`` (rembg) for
        PlantDoc. Routed at the per-image call.
    image_iter
        ``[(rel_path, abs_path), ...]`` for every image to consider.
        ``rel_path`` is what the trainer will pass to
        :func:`mask_path_for`; ``abs_path`` is the actual file to read.

    Returns
    -------
    CacheStats
        Aggregate counts for the per-dataset log.
    """
    out_root = MASK_CACHE_ROOT / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(image_iter)
    cached_already = 0
    newly_segmented = 0
    flagged_count = 0
    failures = 0
    flagged_list: list[dict[str, Any]] = []
    t0 = time.monotonic()

    for i, (rel_path, abs_path) in enumerate(image_iter, start=1):
        mask_out = mask_path_for(dataset, rel_path)
        if mask_out.is_file() and mask_out.stat().st_size > 0:
            cached_already += 1
            if i % log_every == 0:
                _LOGGER.info(
                    "[%s] %d/%d done — %d cached, %d new, %d flagged",
                    dataset, i, total, cached_already, newly_segmented, flagged_count,
                )
            continue
        try:
            flagged, fg = segment_and_cache_one(abs_path, mask_out, style=style)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "[%s] segmentation FAILED on %s: %s — falling back to raw at train time.",
                dataset, rel_path, exc,
            )
            failures += 1
            continue
        newly_segmented += 1
        if flagged:
            flagged_count += 1
            flagged_list.append({
                "rel_path": str(rel_path).replace("\\", "/"),
                "foreground_fraction": float(fg),
            })
        if i % log_every == 0:
            _LOGGER.info(
                "[%s] %d/%d done — %d cached, %d new, %d flagged",
                dataset, i, total, cached_already, newly_segmented, flagged_count,
            )

    elapsed = time.monotonic() - t0
    denom = max(1, newly_segmented)
    stats = CacheStats(
        dataset=dataset,
        style=style,
        total=total,
        cached_already=cached_already,
        newly_segmented=newly_segmented,
        flagged=flagged_count,
        failures=failures,
        flagged_fraction=flagged_count / denom,
        elapsed_seconds=float(elapsed),
    )

    # Persist a per-dataset log so the trainer can read the flagged list.
    log = {
        "dataset": dataset,
        "style": style,
        "total": total,
        "cached_already": cached_already,
        "newly_segmented": newly_segmented,
        "flagged": flagged_count,
        "failures": failures,
        "flagged_fraction": stats.flagged_fraction,
        "elapsed_seconds": stats.elapsed_seconds,
        "flagged_rel_paths": flagged_list,
    }
    dataset_log_path(dataset).write_text(
        json.dumps(log, indent=2), encoding="utf-8",
    )
    _LOGGER.info(
        "[%s] segmentation cache built: total=%d, new=%d, flagged=%d (%.1f%%), failures=%d, elapsed=%.0fs",
        dataset, total, newly_segmented, flagged_count,
        100 * stats.flagged_fraction, failures, elapsed,
    )
    return stats


# --------------------------------------------------------------------- #
# Trainer-time helpers
# --------------------------------------------------------------------- #


def load_flagged_set(dataset: str) -> set[str]:
    """Return the set of rel_paths whose mask was flagged at cache time.

    The randomized trainer uses this to decide: composite (use the
    cached mask) vs fall back to the raw image. POSIX-normalised so
    Windows-built caches read the same as Linux-built ones in Colab.
    """
    log_path = dataset_log_path(dataset)
    if not log_path.is_file():
        return set()
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return {
        str(entry["rel_path"]).replace("\\", "/")
        for entry in data.get("flagged_rel_paths", [])
    }


__all__ = [
    "CacheStats",
    "MASK_CACHE_ROOT",
    "build_mask_cache",
    "dataset_log_path",
    "load_flagged_set",
    "mask_path_for",
    "segment_and_cache_one",
]
