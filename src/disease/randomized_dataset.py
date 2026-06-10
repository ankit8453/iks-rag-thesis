"""HF-row-indexed dataset wrappers for the Phase 5-R retrain.

The original Phase 5 trainer pulls images straight from the published
HF datasets — ``load_dataset("ankit-iiitdmj/iks-plantvillage")`` etc. —
so the trainer works on a fresh Colab runtime with no local data.
Phase 5-R has to follow the same pattern so it can resume after a
free-Colab session timeout and produce its checkpoints to HF.

This module wraps an HF split + the mask cache built by
:mod:`src.disease.segment_cache` and the background pool from
:mod:`src.disease.backgrounds`, and emits ``(image_tensor, label_idx)``
tuples just like the original trainer's ``_HFImageDataset``. Three
modes (chosen at construction):

- ``"randomize"`` — PlantVillage, PlantDoc. Loads ``hf_split[idx]["image"]``,
  the cached mask at ``mask_path_for(dataset_id, split, idx)``, and a
  random :class:`BackgroundEntry` from the pool; composites on the fly.
  If the row was flagged at cache time OR the mask file is missing,
  the row falls back to the raw HF image — a single bad mask never
  blocks a training step.
- ``"raw"`` — Paddy Doctor. Passes the HF image through untouched.
- ``"no_leaf"`` — drawn from a list of (PIL_image, label_idx) tuples
  built from the background pool; powers the PlantDoc-stage 28th
  reject class.

Per-epoch seeded RNG so the same ``(epoch, idx)`` pair reproduces the
same composite — apples-to-apples Grad-CAM comparison between the old
and new checkpoints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.disease.backgrounds import (
    BackgroundEntry,
    composite_leaf_on_bg,
)
from src.disease.segment_cache import is_flagged, mask_path_for
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)

Mode = Literal["randomize", "raw", "no_leaf"]


@dataclass
class NoLeafRow:
    """One row of the ``no_leaf`` reject class.

    Backed by an on-disk JPEG (from the background pool's cache) — the
    dataset opens it lazily so a large no-leaf pool doesn't blow up
    memory.
    """

    abs_path: Path
    label_idx: int


# --------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------- #


class HFRandomizedDiseaseDataset:
    """torch.utils.data.Dataset-compatible HF wrapper with on-the-fly
    background randomization.

    Parameters
    ----------
    hf_split
        A loaded HF dataset split (already :func:`load_dataset` ed by
        the caller). Yields rows with ``"image"`` and ``"label_idx"``.
    dataset_id
        Local identifier used to find cached masks — must match what
        was passed to :func:`~src.disease.segment_cache.build_mask_cache_from_hf`.
    split
        ``"train"`` / ``"val"`` / ``"test"``. Lets the wrapper find the
        right mask subdirectory and the flagged set.
    mode
        Per-row treatment — see module docstring.
    bg_pool
        :class:`BackgroundEntry` list. Only used in ``"randomize"`` mode.
    transform
        Optional albumentations pipeline. Applied AFTER compositing or
        pass-through so the augmentation operates on the final RGB.
    no_leaf_rows
        Rows for the ``"no_leaf"`` mode. Appended to the HF rows so
        ``__len__`` is ``len(hf_split) + len(no_leaf_rows)``. Indices
        ``< len(hf_split)`` are HF rows; indices ``>=`` are no-leaf
        rows. Ignored unless ``mode == "no_leaf"`` OR
        ``mode == "randomize"`` (PlantDoc stage adds no-leaf to the
        train split that way).
    seed
        Base seed for the per-epoch RNG.
    """

    def __init__(
        self,
        hf_split: Any,
        *,
        dataset_id: str,
        split: str,
        mode: Mode,
        bg_pool: list[BackgroundEntry] | None = None,
        transform: Any | None = None,
        no_leaf_rows: list[NoLeafRow] | None = None,
        seed: int = 42,
    ) -> None:
        self.hf_split = hf_split
        self.dataset_id = dataset_id
        self.split = split
        self.mode: Mode = mode
        self.bg_pool = list(bg_pool or [])
        self.transform = transform
        self.no_leaf_rows = list(no_leaf_rows or [])
        self.seed = int(seed)
        self._epoch = 0

        # Pre-load the flagged set once so __getitem__ is fast.
        if mode == "randomize":
            from src.disease.segment_cache import load_flagged_set  # noqa: PLC0415
            self._flagged_set = load_flagged_set(dataset_id)
        else:
            self._flagged_set = set()

    # ----- epoch-aware RNG ------------------------------------------ #

    def set_epoch(self, epoch: int) -> None:
        """Trainer calls this once per epoch to reseed the bg RNG."""
        self._epoch = int(epoch)

    def _rng_for(self, idx: int) -> random.Random:
        return random.Random((self.seed * 1_000_003) + (self._epoch * 7919) + idx)

    # ----- length + indexing --------------------------------------- #

    def __len__(self) -> int:
        return len(self.hf_split) + len(self.no_leaf_rows)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415

        n_hf = len(self.hf_split)
        if idx >= n_hf:
            # ---------- no_leaf row (raw, reject label) ----------
            entry = self.no_leaf_rows[idx - n_hf]
            with PILImage.open(entry.abs_path) as src:
                arr = np.asarray(src.convert("RGB"))
            if self.transform is not None:
                arr = self.transform(image=arr)["image"]
            return arr, int(entry.label_idx)

        # ---------- HF row ----------
        row = self.hf_split[idx]
        pil = row["image"].convert("RGB")
        label = int(row["label_idx"])

        if self.mode == "raw":
            arr = np.asarray(pil)
        elif self.mode == "randomize":
            # Flagged or missing mask → fall back to raw, never crash.
            key = f"{self.split}/{int(idx):06d}"
            use_raw = key in self._flagged_set
            mask_p = mask_path_for(self.dataset_id, self.split, idx)
            if not mask_p.is_file():
                use_raw = True
            if use_raw or not self.bg_pool:
                arr = np.asarray(pil)
            else:
                with PILImage.open(mask_p) as m:
                    mask = m.convert("L")
                rng = self._rng_for(idx)
                bg_entry = rng.choice(self.bg_pool)
                composed = composite_leaf_on_bg(
                    pil, mask, bg_entry.path,
                    out_size=pil.size, rng=rng,
                )
                arr = np.asarray(composed)
        else:
            raise ValueError(f"unknown HF dataset mode: {self.mode!r}")

        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        return arr, label


# --------------------------------------------------------------------- #
# no_leaf row builder
# --------------------------------------------------------------------- #


def build_no_leaf_rows(
    bg_pool: list[BackgroundEntry],
    label_idx: int,
    *,
    sources: tuple[str, ...] = ("pandey_background",),
    max_n: int | None = None,
) -> list[NoLeafRow]:
    """Convert a slice of the background pool into ``NoLeafRow``s.

    Default behaviour: pick only ``pandey_background`` entries (the
    only true "not a leaf" source — phantomfs / sirajganj are real soil
    that the model legitimately needs to know about). On Colab where
    Pandey isn't available, the caller can pass ``sources=("phantomfs",
    "sirajganj")`` to bootstrap a smaller no-leaf class from soil
    images that genuinely don't contain a leaf either.
    """
    rows: list[NoLeafRow] = []
    for entry in bg_pool:
        if entry.source not in sources:
            continue
        rows.append(NoLeafRow(abs_path=entry.path, label_idx=int(label_idx)))
        if max_n is not None and len(rows) >= max_n:
            break
    _LOGGER.info(
        "Built %d no_leaf rows from sources=%s",
        len(rows), sources,
    )
    return rows


# --------------------------------------------------------------------- #
# Convenience: load HF split + wrap
# --------------------------------------------------------------------- #


def load_hf_randomized(
    *,
    dataset_repo: str,
    dataset_id: str,
    split: str,
    mode: Mode,
    transform: Any | None,
    bg_pool: list[BackgroundEntry] | None = None,
    no_leaf_rows: list[NoLeafRow] | None = None,
    seed: int = 42,
) -> HFRandomizedDiseaseDataset:
    """One-call helper used by the cascade trainer.

    Pulls the HF split via :func:`datasets.load_dataset`, then wraps
    it as an :class:`HFRandomizedDiseaseDataset`.
    """
    from datasets import load_dataset  # noqa: PLC0415

    hf = load_dataset(dataset_repo, split=split)
    return HFRandomizedDiseaseDataset(
        hf_split=hf, dataset_id=dataset_id, split=split, mode=mode,
        bg_pool=bg_pool, transform=transform,
        no_leaf_rows=no_leaf_rows, seed=seed,
    )


__all__ = [
    "HFRandomizedDiseaseDataset",
    "Mode",
    "NoLeafRow",
    "build_no_leaf_rows",
    "load_hf_randomized",
]
