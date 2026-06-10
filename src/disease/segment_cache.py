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
import tarfile
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

# HF Hub dataset repo used to back up the local mask cache so a Colab
# session-timeout (or a switch to a second account) doesn't lose the
# segmentation work. One ``.tar.gz`` per dataset (PV / PD), pushed
# every N rows AND at end-of-split. Same resume-friendly contract as
# the trainer's ``CheckpointManager``.
DEFAULT_MASK_BACKUP_REPO: str = "ankit-iiitdmj/iks-disease-r-mask-cache"


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


def _short(p: Path) -> str:
    """Best-effort repo-relative path for log messages — falls back to
    the absolute path when ``p`` is outside :data:`PROJECT_ROOT`
    (happens in tests where the cache root is monkeypatched into a
    tmp dir)."""
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


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
    hf_backup_repo: str | None = DEFAULT_MASK_BACKUP_REPO,
    hf_push_every_n_rows: int = 5000,
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
    hf_backup_repo
        HF Hub dataset repo to back the mask cache up to. ``None``
        disables backup entirely (laptop dev). Default is the locked
        :data:`DEFAULT_MASK_BACKUP_REPO` -- a private dataset repo
        under the same HF account that owns the model checkpoints. At
        startup the function pulls any existing backup (so a fresh
        Colab runtime resumes from where the previous session
        stopped); during the run it pushes every
        ``hf_push_every_n_rows`` newly segmented rows; at end-of-split
        it pushes one final time. Pushes survive a session timeout.
    hf_push_every_n_rows
        Interval (in newly-segmented rows, NOT in total) between HF
        pushes during a run. Default 5000 ⇒ at most ~5000 rows of
        compute lost on a session death; the upload itself takes ~1 min
        per push for a 1 GB tarball, so 5000 is a sensible tradeoff.
    """
    from datasets import load_dataset  # noqa: PLC0415

    # ----- resume from HF if a backup exists ----------------------- #
    # Pull is idempotent: if the local cache is already complete, this
    # is a no-op; if it's partial, the missing files come back. The
    # ``cached_already`` counter below then correctly counts everything
    # the pull provided plus everything segmented in earlier sessions.
    if hf_backup_repo:
        pull_mask_cache_from_hf(dataset_id, repo_id=hf_backup_repo)

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

        # ---- periodic HF push -------------------------------------- #
        # Save the log + push the tarball every `hf_push_every_n_rows`
        # newly segmented rows. A Colab session timeout between two
        # pushes only loses up to that many rows of work.
        if (
            hf_backup_repo
            and newly_segmented > 0
            and newly_segmented % hf_push_every_n_rows == 0
        ):
            _LOGGER.info(
                "[%s/%s] periodic HF backup at %d new rows ...",
                dataset_id, split, newly_segmented,
            )
            _persist_log_so_far(
                dataset_id=dataset_id, split=split, style=style,
                total=total, cached_already=cached_already,
                newly_segmented=newly_segmented,
                flagged_count=flagged_count, failures=failures,
                elapsed_seconds=time.monotonic() - t0,
                flagged_keys_this_split=flagged_keys_this_split,
            )
            try:
                push_mask_cache_to_hf(dataset_id, repo_id=hf_backup_repo)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "Periodic HF push for %s failed: %s -- continuing; "
                    "next push will retry.",
                    dataset_id, exc,
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

    _persist_log_so_far(
        dataset_id=dataset_id, split=split, style=style,
        total=total, cached_already=cached_already,
        newly_segmented=newly_segmented,
        flagged_count=flagged_count, failures=failures,
        elapsed_seconds=elapsed,
        flagged_keys_this_split=flagged_keys_this_split,
    )

    # ---- final end-of-split HF push ------------------------------- #
    # Always push at end-of-split, even when no new rows were
    # segmented (the log itself may have changed because cached_already
    # incremented). Wrapped so an HF outage at the very end doesn't
    # wipe the local cache stats.
    if hf_backup_repo and newly_segmented > 0:
        try:
            push_mask_cache_to_hf(dataset_id, repo_id=hf_backup_repo)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "Final HF push for %s failed: %s -- local cache is "
                "intact; re-run to retry.",
                dataset_id, exc,
            )

    _LOGGER.info(
        "[%s/%s] cache built: total=%d, new=%d, flagged=%d (%.1f%%), failures=%d, elapsed=%.0fs",
        dataset_id, split, total, newly_segmented, flagged_count,
        100 * stats.flagged_fraction, failures, elapsed,
    )
    return stats


def _persist_log_so_far(
    *,
    dataset_id: str, split: str, style: str,
    total: int, cached_already: int, newly_segmented: int,
    flagged_count: int, failures: int,
    elapsed_seconds: float,
    flagged_keys_this_split: list[str],
) -> None:
    """Write the per-dataset segmentation log to disk.

    Factored out of :func:`build_mask_cache_from_hf` so the periodic
    HF push can call it too -- otherwise a session death between
    pushes would leave the log out of sync with the masks already
    written.
    """
    denom = max(1, newly_segmented)
    log = _load_existing_log(dataset_id)
    log["splits"][split] = {
        "style": style, "total": total,
        "cached_already": cached_already,
        "newly_segmented": newly_segmented,
        "flagged": flagged_count,
        "failures": failures,
        "flagged_fraction": flagged_count / denom,
        "elapsed_seconds": float(elapsed_seconds),
    }
    union = set(log.get("flagged_keys") or [])
    union.update(flagged_keys_this_split)
    log["flagged_keys"] = sorted(union)
    _save_log(dataset_id, log)


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


# --------------------------------------------------------------------- #
# HF Hub backup / restore for the mask cache
# --------------------------------------------------------------------- #
#
# Why this exists: Colab `/content/` is ephemeral. A free-tier session
# timeout mid-segmentation would otherwise lose every mask written so
# far -- 17 000 PlantVillage masks at ~50 KB each = ~850 MB of work.
# Same resume contract the trainer's CheckpointManager has, but for
# the segmentation pass.
#
# Layout on HF (one private dataset repo, one tar.gz per dataset):
#
#     ankit-iiitdmj/iks-disease-r-mask-cache/
#     ├── plantvillage.tar.gz   (contains plantvillage/{train,val,test}/*.png)
#     └── plantdoc.tar.gz       (contains plantdoc/{train,val,test}/*.png)
#
# Each tar is created with the dataset_id as the inner directory name
# so untar-into-MASK_CACHE_ROOT lands the files at the exact path the
# trainer expects.


def push_mask_cache_to_hf(
    dataset_id: str,
    *,
    repo_id: str = DEFAULT_MASK_BACKUP_REPO,
    private: bool = True,
) -> Path | None:
    """Tar+gz the local mask cache for ``dataset_id`` and upload to HF.

    Replaces the previous ``<dataset_id>.tar.gz`` on the repo (HF Hub
    handles overwrite atomically via commits, so a partial upload can't
    leave the repo in a half-state). The HF dataset repo is created
    with ``private=True`` if it doesn't exist yet.

    Returns the path to the local tar (deleted after upload), or
    ``None`` when there are no local masks to push.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    local_dir = MASK_CACHE_ROOT / dataset_id
    png_files = (
        list(local_dir.rglob("*.png")) if local_dir.is_dir() else []
    )
    if not png_files:
        _LOGGER.info("No local masks for %s; nothing to push.", dataset_id)
        return None

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset",
        private=private, exist_ok=True,
    )

    tar_path = MASK_CACHE_ROOT / f"_backup_{dataset_id}.tar.gz"
    _LOGGER.info(
        "Tarring %d mask file(s) for %s -> %s ...",
        len(png_files), dataset_id,
        _short(tar_path),
    )
    # arcname=dataset_id so untar reproduces "<dataset_id>/<split>/<idx>.png"
    # relative to MASK_CACHE_ROOT -- matches the path the trainer reads.
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tar:
        # Add the log first so a partial extraction has the keys.
        log = dataset_log_path(dataset_id)
        if log.is_file():
            tar.add(log, arcname=f"{dataset_id}/{_LOG_FILENAME}")
        # Then per-split subtrees.
        for split_dir in sorted(local_dir.iterdir()):
            if split_dir.is_dir():
                tar.add(split_dir, arcname=f"{dataset_id}/{split_dir.name}")
    size_mb = tar_path.stat().st_size / 1024**2

    _LOGGER.info(
        "Uploading %.1f MB tarball for %s to %s ...",
        size_mb, dataset_id, repo_id,
    )
    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo=f"{dataset_id}.tar.gz",
        repo_id=repo_id, repo_type="dataset",
    )
    _LOGGER.info(
        "Pushed %s.tar.gz (%.1f MB, %d masks) to %s",
        dataset_id, size_mb, len(png_files), repo_id,
    )
    try:
        tar_path.unlink()
    except OSError:
        pass
    return tar_path


def pull_mask_cache_from_hf(
    dataset_id: str,
    *,
    repo_id: str = DEFAULT_MASK_BACKUP_REPO,
) -> bool:
    """Download ``<dataset_id>.tar.gz`` from HF and extract under
    :data:`MASK_CACHE_ROOT`.

    Returns ``True`` if a backup was pulled (some masks may now be
    present locally), ``False`` if no backup exists yet (first run for
    this dataset).

    Idempotent: tar extraction overwrites existing files with identical
    content (same row → same mask), so re-pulling is safe.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import (  # noqa: PLC0415
        EntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        local_tar = hf_hub_download(
            repo_id=repo_id,
            filename=f"{dataset_id}.tar.gz",
            repo_type="dataset",
        )
    except (RepositoryNotFoundError, EntryNotFoundError):
        _LOGGER.info(
            "No HF backup for %s in %s yet; starting fresh.",
            dataset_id, repo_id,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "HF pull for %s failed: %s; starting fresh.",
            dataset_id, exc,
        )
        return False

    _LOGGER.info(
        "Pulled %s backup from %s; extracting into %s ...",
        dataset_id, repo_id,
        _short(MASK_CACHE_ROOT),
    )
    MASK_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(MASK_CACHE_ROOT)
    n_files = len(list((MASK_CACHE_ROOT / dataset_id).rglob("*.png")))
    _LOGGER.info(
        "Extraction complete for %s: %d masks locally.",
        dataset_id, n_files,
    )
    return True


__all__ = [
    "CacheStats",
    "DEFAULT_MASK_BACKUP_REPO",
    "MASK_CACHE_ROOT",
    "build_mask_cache_from_hf",
    "dataset_log_path",
    "is_flagged",
    "load_flagged_set",
    "mask_path_for",
    "pull_mask_cache_from_hf",
    "push_mask_cache_to_hf",
    "segment_and_cache_one_image",
]
