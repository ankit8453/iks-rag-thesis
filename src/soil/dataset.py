"""Soil dataset wrappers (Phase 4 §G).

Re-uses :class:`~src.disease.dataset.JSONIndexedImageDataset` instead of
duplicating it; the underlying split-JSON layout is identical.

Three factory functions:
- :func:`make_phantomfs_loaders` — Phantom-fs (7 deposit classes).
- :func:`make_irsid_loaders` — IRSID (Kaggle mirror, 3 texture classes,
  16 samples). Returns only a test loader (IRSID is test-only under §D).
- :func:`make_soil_cross_region_loaders` — the §14 cross-region setup:
  Phantom-fs train+val + IRSID test, with the cross-region split JSONs
  whose paths are prefixed with ``phantomfs/`` or ``irsid/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.disease.dataset import JSONIndexedImageDataset
from src.soil.config import SoilConfig
from src.utils.data_stats import load_channel_stats
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_SOIL_DIR, PROJECT_ROOT

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)

PHANTOMFS_DEFAULT_ROOT: Path = DATA_SOIL_DIR / "phantomfs" / "raw"
IRSID_DEFAULT_ROOT: Path = DATA_SOIL_DIR / "irsid" / "raw"
SIRAJGANJ_DEFAULT_ROOT: Path = DATA_SOIL_DIR / "sirajganj_moisture" / "raw"
SOIL_TYPES_DEFAULT_ROOT: Path = PHANTOMFS_DEFAULT_ROOT  # legacy alias

_SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
_CONFIGS_DATA = PROJECT_ROOT / "configs" / "data"


@dataclass
class SoilSample:
    """One soil image plus its multi-task labels and provenance."""

    image_path: Path
    soil_type: str
    texture: str
    surface: str
    moisture_appearance: str
    cover_state: str
    region: str | None = None


class SoilTypeDataset(JSONIndexedImageDataset):
    """Backwards-compatible alias."""


def _build_simple_loaders(
    dataset_name: str,
    raw_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.soil.transforms import (  # noqa: PLC0415
        build_soil_eval_aug,
        build_soil_train_aug,
    )

    stats = load_channel_stats(_CONFIGS_DATA / f"{dataset_name}_norm.yaml")
    train_aug = build_soil_train_aug(image_size, stats.mean, stats.std)
    eval_aug = build_soil_eval_aug(image_size, stats.mean, stats.std)

    splits_dir = _SPLITS_ROOT / dataset_name
    loaders: dict[str, DataLoader] = {}
    for split_name, transform in (
        ("train", train_aug),
        ("val", eval_aug),
        ("test", eval_aug),
    ):
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.is_file():
            _LOGGER.info(
                "No %s split for %s; skipping that loader.", split_name, dataset_name
            )
            continue
        ds = JSONIndexedImageDataset(split_path, raw_root, transform=transform)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def make_phantomfs_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: SoilConfig | None = None,
) -> dict[str, Any]:
    """train/val/test loaders for the Phantom-fs primary soil dataset."""
    return _build_simple_loaders(
        "phantomfs", PHANTOMFS_DEFAULT_ROOT, 224, batch_size, num_workers
    )


def make_sirajganj_moisture_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: SoilConfig | None = None,
) -> dict[str, Any]:
    """train/val/test loaders for the Sirajganj moisture dataset [ADDED].

    Supervises the soil module's ``moisture_appearance`` head per master
    plan §14 (``dry`` / ``moderate`` / ``wet``). Underlying images live
    at ``data/soil/sirajganj_moisture/raw/Soil_Moisture_Dataset/Before
    Augmentation/<class>/``; the ``After Augmentation`` author-pre-
    augmented copies are deferred (we augment ourselves at training time
    via :mod:`src.soil.transforms`). The split JSON's ``path`` entries
    already encode the ``Soil_Moisture_Dataset/Before Augmentation/...``
    prefix, so the loader's raw root is the dataset's top-level
    ``raw/`` directory.
    """
    return _build_simple_loaders(
        "sirajganj_moisture",
        SIRAJGANJ_DEFAULT_ROOT,
        224,
        batch_size,
        num_workers,
    )


def make_irsid_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: SoilConfig | None = None,
) -> dict[str, Any]:
    """IRSID standalone loader — returns ``{"test": loader}``.

    Loads the cross-region test split (which is 100% IRSID) under the
    cross-region prefixed-path resolution. Phase 6 will more typically
    use :func:`make_soil_cross_region_loaders` instead.
    """
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.soil.transforms import build_soil_eval_aug  # noqa: PLC0415

    stats = load_channel_stats(_CONFIGS_DATA / "irsid_norm.yaml")
    eval_aug = build_soil_eval_aug(224, stats.mean, stats.std)

    split_path = _SPLITS_ROOT / "soil_cross_region" / "test.json"
    ds = _PrefixedCrossRegionDataset(split_path, DATA_SOIL_DIR, transform=eval_aug)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return {"test": loader}


def make_soil_cross_region_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: SoilConfig | None = None,
) -> dict[str, Any]:
    """§14 cross-region loaders: Phantom-fs train+val + IRSID test."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.soil.transforms import (  # noqa: PLC0415
        build_soil_eval_aug,
        build_soil_train_aug,
    )

    stats = load_channel_stats(_CONFIGS_DATA / "phantomfs_norm.yaml")
    train_aug = build_soil_train_aug(224, stats.mean, stats.std)
    eval_aug = build_soil_eval_aug(224, stats.mean, stats.std)

    splits_dir = _SPLITS_ROOT / "soil_cross_region"
    shared_root = DATA_SOIL_DIR
    return {
        "train": DataLoader(
            _PrefixedCrossRegionDataset(
                splits_dir / "train.json", shared_root, train_aug
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            _PrefixedCrossRegionDataset(
                splits_dir / "val.json", shared_root, eval_aug
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            _PrefixedCrossRegionDataset(
                splits_dir / "test.json", shared_root, eval_aug
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


class _PrefixedCrossRegionDataset(JSONIndexedImageDataset):
    """Resolve cross-region paths of the form ``<dataset>/<rest>``.

    Each ``SplitEntry.path`` begins with ``phantomfs/`` or ``irsid/``.
    The actual file is at ``<shared_root>/<dataset>/raw/<rest>``.
    """

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        entry = self.entries[idx]
        head, rest = entry.path.split("/", 1)
        image_path = self.raw_root / head / "raw" / rest
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        return arr, entry.label_idx


__all__ = [
    "IRSID_DEFAULT_ROOT",
    "PHANTOMFS_DEFAULT_ROOT",
    "SOIL_TYPES_DEFAULT_ROOT",
    "SoilSample",
    "SoilTypeDataset",
    "make_irsid_loaders",
    "make_phantomfs_loaders",
    "make_soil_cross_region_loaders",
]
