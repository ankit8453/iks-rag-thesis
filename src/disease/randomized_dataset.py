"""Dataset wrappers for the Phase 5-R background-randomization retrain.

Three modes, picked per-dataset:

- ``"randomize"`` (PlantVillage, PlantDoc) — load image + cached mask +
  a random background from the pool, composite on-the-fly, return the
  composited image. Background is re-rolled every ``__getitem__`` call
  so a single image sees many backgrounds across epochs and the model
  can't latch onto the background as a label cue.
- ``"raw"`` (Paddy Doctor) — pass through the original image untouched.
  Phase 5-R Part 1 found Paddy is full-canopy field photos with no
  meaningful foreground/background split, so no randomization is
  applied. The class label is preserved.
- ``"no_leaf"`` (Pandey ``Background_without_leaves`` slice +
  bare-soil hold-out) — return the raw background image with the
  reject-class label. Powers the PlantDoc-stage no-leaf reject head.

The trainer wraps one of these per stage and treats them as a single
``torch.utils.data.Dataset`` from then on.

Deterministic per-epoch seeding so a re-run on the same epoch produces
the same composites for the same images — important for reproducible
Grad-CAM comparisons later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.disease.backgrounds import (
    BackgroundEntry,
    build_background_pool,
    composite_leaf_on_bg,
)
from src.disease.segment_cache import load_flagged_set, mask_path_for
from src.utils.data_splits import load_class_map, load_split
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)

Mode = Literal["randomize", "raw", "no_leaf"]


@dataclass
class SampleSpec:
    """One row in the dataset.

    Attributes
    ----------
    rel_path : str
        Path under ``raw_root`` (POSIX-style for cache-key stability).
    abs_path : Path
        Absolute path to the image on disk.
    label_idx : int
        Integer class id this sample carries.
    mode : Mode
        How to load the sample — see module docstring.
    """

    rel_path: str
    abs_path: Path
    label_idx: int
    mode: Mode


# --------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------- #


class RandomizedDiseaseDataset:
    """``torch.utils.data.Dataset``-compatible random-bg compositor.

    Parameters
    ----------
    samples
        Pre-built list of :class:`SampleSpec`. Builders below take
        care of constructing it for each cascade stage.
    dataset_id
        Used only to read the cached mask via
        :func:`~src.disease.segment_cache.mask_path_for`. Pass the same
        identifier the cache was built with (``"plantvillage"`` /
        ``"plantdoc"``).
    bg_pool
        List of :class:`BackgroundEntry`. The "randomize" mode samples
        from it; "raw" and "no_leaf" ignore it.
    transform
        Optional albumentations pipeline. Applied AFTER compositing /
        pass-through so the augmentation operates on the final RGB the
        model will see.
    seed
        Base seed for the per-epoch RNG. Combined with the current epoch
        (see :meth:`set_epoch`) for reproducibility.
    flagged_rel_paths
        Pre-loaded set of rel_paths whose mask failed the 5–95 % guard
        at cache time — those rows fall back to raw at train time.
    """

    def __init__(
        self,
        samples: list[SampleSpec],
        *,
        dataset_id: str,
        bg_pool: list[BackgroundEntry],
        transform: Any | None = None,
        seed: int = 42,
        flagged_rel_paths: set[str] | None = None,
    ) -> None:
        self.samples = samples
        self.dataset_id = dataset_id
        self.bg_pool = bg_pool
        self.transform = transform
        self.seed = int(seed)
        self.flagged_rel_paths = flagged_rel_paths or set()
        self._epoch = 0

    # ----- epoch-aware RNG ------------------------------------------ #

    def set_epoch(self, epoch: int) -> None:
        """Trainer calls this once per epoch to reseed the bg RNG."""
        self._epoch = int(epoch)

    def _rng_for(self, idx: int) -> random.Random:
        # Index-aware so workers can shard cleanly; epoch-aware so each
        # epoch sees a different background per image.
        return random.Random((self.seed * 1_000_003) + (self._epoch * 7919) + idx)

    # ----- core load + composite ------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415

        spec = self.samples[idx]
        with PILImage.open(spec.abs_path) as src:
            src = src.convert("RGB")

            # ---------- mode dispatch ----------
            if spec.mode == "raw" or spec.mode == "no_leaf":
                arr = np.asarray(src)
            elif spec.mode == "randomize":
                # If this row was flagged at cache time OR mask file
                # missing, fall back to raw — never block a train step.
                use_raw = spec.rel_path in self.flagged_rel_paths
                mask_p = mask_path_for(self.dataset_id, spec.rel_path)
                if not mask_p.is_file():
                    use_raw = True
                if use_raw or not self.bg_pool:
                    arr = np.asarray(src)
                else:
                    with PILImage.open(mask_p) as m:
                        mask = m.convert("L")
                    rng = self._rng_for(idx)
                    bg_entry = rng.choice(self.bg_pool)
                    composed = composite_leaf_on_bg(
                        src, mask, bg_entry.path,
                        out_size=src.size, rng=rng,
                    )
                    arr = np.asarray(composed)
            else:
                raise ValueError(f"unknown sample mode: {spec.mode!r}")

        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        return arr, int(spec.label_idx)


# --------------------------------------------------------------------- #
# Per-stage sample builders
# --------------------------------------------------------------------- #


def build_samples_from_split(
    split_path: Path,
    raw_root: Path,
    *,
    mode: Mode,
    label_offset: int = 0,
) -> list[SampleSpec]:
    """Read a Phase 4 split JSON and produce :class:`SampleSpec` rows."""
    entries = load_split(split_path)
    out: list[SampleSpec] = []
    for entry in entries:
        rel = str(entry.path).replace("\\", "/")
        abs_p = (raw_root / entry.path)
        out.append(SampleSpec(
            rel_path=rel, abs_path=abs_p,
            label_idx=int(entry.label_idx) + int(label_offset),
            mode=mode,
        ))
    return out


def build_no_leaf_samples(
    sources: list[Path],
    label_idx: int,
    *,
    max_per_source: int | None = None,
) -> list[SampleSpec]:
    """Walk one or more directories of "not a leaf" backgrounds and
    return raw samples with the reject label.

    Used by the PlantDoc stage to bring in the 28th class.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    out: list[SampleSpec] = []
    for root in sources:
        root = Path(root)
        if not root.is_dir():
            _LOGGER.warning("no_leaf source missing: %s", root)
            continue
        found = 0
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in img_exts:
                rel = str(p.relative_to(root)).replace("\\", "/")
                out.append(SampleSpec(
                    rel_path=f"no_leaf/{root.name}/{rel}",
                    abs_path=p,
                    label_idx=int(label_idx),
                    mode="no_leaf",
                ))
                found += 1
                if max_per_source is not None and found >= max_per_source:
                    break
        _LOGGER.info("no_leaf source %s contributed %d sample(s).", root.name, found)
    return out


# --------------------------------------------------------------------- #
# Convenience: build a randomized dataset from a Phase 4 split
# --------------------------------------------------------------------- #


def make_randomized_dataset(
    *,
    split_path: Path,
    raw_root: Path,
    dataset_id: str,
    transform: Any | None,
    seed: int = 42,
    mode: Mode = "randomize",
    extra_samples: list[SampleSpec] | None = None,
    bg_pool: list[BackgroundEntry] | None = None,
) -> RandomizedDiseaseDataset:
    """One-call helper used by the cascade trainer.

    - Builds samples from the split JSON in the requested ``mode``.
    - Optionally appends ``extra_samples`` (e.g. no-leaf rows at the
      PlantDoc stage).
    - Constructs (or accepts) the background pool.
    - Loads the flagged-rel-paths set so flagged rows fall back to raw.
    """
    samples = build_samples_from_split(
        split_path=split_path, raw_root=raw_root, mode=mode,
    )
    if extra_samples:
        samples = samples + list(extra_samples)
    pool = bg_pool if bg_pool is not None else build_background_pool()
    flagged = load_flagged_set(dataset_id) if mode == "randomize" else set()
    return RandomizedDiseaseDataset(
        samples=samples,
        dataset_id=dataset_id,
        bg_pool=pool,
        transform=transform,
        seed=seed,
        flagged_rel_paths=flagged,
    )


# --------------------------------------------------------------------- #
# Class-map utility
# --------------------------------------------------------------------- #


def load_class_map_with_no_leaf(
    class_map_path: Path,
    no_leaf_label: str = "no_leaf",
) -> tuple[dict[str, int], int]:
    """Load a ``class_map.json`` and append the no-leaf reject class.

    Returns ``(extended_class_map, no_leaf_idx)``. The no-leaf class is
    appended at index ``max(existing) + 1`` so existing label ids do
    NOT shift — the cascade transfer is safe.
    """
    base = load_class_map(class_map_path)
    if no_leaf_label in base:
        return dict(base), int(base[no_leaf_label])
    next_idx = max(base.values()) + 1
    extended = dict(base)
    extended[no_leaf_label] = int(next_idx)
    return extended, int(next_idx)


__all__ = [
    "Mode",
    "RandomizedDiseaseDataset",
    "SampleSpec",
    "build_no_leaf_samples",
    "build_samples_from_split",
    "load_class_map_with_no_leaf",
    "make_randomized_dataset",
]
