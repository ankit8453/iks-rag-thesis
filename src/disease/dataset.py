"""Dataset wrappers for plant disease training and evaluation (Phase 4 §G).

Per master reference §41 the image datasets live under the top-level
``data/`` directory, not inside ``corpus/``. Split JSONs live under
``data/splits/<dataset>/``.

The generic :class:`JSONIndexedImageDataset` reads a Phase 4 split JSON
(``train.json`` / ``val.json`` / ``test.json``) plus a sibling
``class_map.json`` and returns ``(image_tensor, label_idx)`` tuples
suitable for ``torch.utils.data.DataLoader``.

Three factory functions wire up the per-dataset loaders for
PlantVillage, PlantDoc, and Paddy Doctor using the right raw root and
image size from each ``configs/data/<dataset>_norm.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.disease.config import DiseaseConfig
from src.utils.data_splits import load_class_map, load_split
from src.utils.data_stats import load_channel_stats
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PLANT_DISEASE_DIR, PROJECT_ROOT

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)

PLANTVILLAGE_DEFAULT_ROOT: Path = DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw"
PLANTDOC_DEFAULT_ROOT: Path = DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw"
PADDY_DOCTOR_DEFAULT_ROOT: Path = DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw"

_SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
_CONFIGS_DATA = PROJECT_ROOT / "configs" / "data"


class JSONIndexedImageDataset:
    """Reads a Phase 4 split JSON and returns ``(image, label_idx)`` items.

    Lives in :mod:`src.disease` but is fully generic — the soil module
    re-imports the same class rather than duplicating.

    Parameters
    ----------
    split_path : Path
        Path to a ``train.json`` / ``val.json`` / ``test.json`` written
        by :func:`src.utils.data_splits.save_split`.
    raw_root : Path
        Dataset's raw root; ``SplitEntry.path`` is joined onto this.
    transform : albumentations.Compose | None
        Albumentations pipeline returning a ``CHW float32`` torch
        tensor. Pass ``None`` to skip transforms (useful in unit
        tests).
    """

    def __init__(
        self,
        split_path: Path,
        raw_root: Path,
        transform: Any | None = None,
    ) -> None:
        self.entries = load_split(split_path)
        self.raw_root = Path(raw_root)
        self.transform = transform

        cm_path = split_path.parent / "class_map.json"
        if cm_path.is_file():
            self.class_map = load_class_map(cm_path)
        else:
            labels = sorted({e.label for e in self.entries})
            self.class_map = {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        entry = self.entries[idx]
        image_path = self.raw_root / entry.path
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        return arr, entry.label_idx


def _build_loaders(
    dataset_name: str,
    raw_root: Path,
    image_size: int,
    config: DiseaseConfig | None,
    batch_size: int,
    num_workers: int,
) -> dict[str, "DataLoader"]:
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.disease.transforms import (  # noqa: PLC0415
        build_disease_eval_aug,
        build_disease_train_aug,
    )

    stats = load_channel_stats(_CONFIGS_DATA / f"{dataset_name}_norm.yaml")
    train_aug = build_disease_train_aug(image_size, stats.mean, stats.std)
    eval_aug = build_disease_eval_aug(image_size, stats.mean, stats.std)

    splits_dir = _SPLITS_ROOT / dataset_name
    loaders: dict[str, DataLoader] = {}
    for split_name, transform in (
        ("train", train_aug),
        ("val", eval_aug),
        ("test", eval_aug),
    ):
        ds = JSONIndexedImageDataset(
            splits_dir / f"{split_name}.json",
            raw_root,
            transform=transform,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    _LOGGER.info(
        "Built %s loaders: train=%d, val=%d, test=%d",
        dataset_name,
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
    )
    return loaders


def make_plantvillage_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: DiseaseConfig | None = None,
) -> dict[str, "DataLoader"]:
    """train/val/test DataLoaders for PlantVillage (380×380, EfficientNet-B4)."""
    return _build_loaders(
        "plantvillage", PLANTVILLAGE_DEFAULT_ROOT, 380, config, batch_size, num_workers
    )


def make_plantdoc_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: DiseaseConfig | None = None,
) -> dict[str, "DataLoader"]:
    """train/val/test DataLoaders for PlantDoc (380×380)."""
    return _build_loaders(
        "plantdoc", PLANTDOC_DEFAULT_ROOT, 380, config, batch_size, num_workers
    )


def make_paddy_doctor_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    config: DiseaseConfig | None = None,
) -> dict[str, "DataLoader"]:
    """train/val/test DataLoaders for Paddy Doctor (380×380)."""
    return _build_loaders(
        "paddy_doctor",
        PADDY_DOCTOR_DEFAULT_ROOT,
        380,
        config,
        batch_size,
        num_workers,
    )


class PlantVillageDataset(JSONIndexedImageDataset):
    """Backwards-compatible alias — same as :class:`JSONIndexedImageDataset`."""


class PlantDocDataset(JSONIndexedImageDataset):
    """Backwards-compatible alias — same as :class:`JSONIndexedImageDataset`."""


__all__ = [
    "JSONIndexedImageDataset",
    "PADDY_DOCTOR_DEFAULT_ROOT",
    "PLANTDOC_DEFAULT_ROOT",
    "PLANTVILLAGE_DEFAULT_ROOT",
    "PlantDocDataset",
    "PlantVillageDataset",
    "make_paddy_doctor_loaders",
    "make_plantdoc_loaders",
    "make_plantvillage_loaders",
]
