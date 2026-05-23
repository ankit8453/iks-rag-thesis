"""OLID I multi-label dataset wrapper (Phase 4 §G, [ADDED] for C5).

OLID I is multi-label — a single leaf can carry both a deficiency tag
and a pest tag. Causation-related labels here are **used as ground
truth for the C5 evaluation in Phase 11**, NOT predicted from images by
Phase 5 vision modules. Per master reference §14, the system does not
infer causation from images.

Phase 4 ships a smoke-sample of OLID I (bottle gourd only). The
multi-label structure here is induced from the folder name: each image
sits in exactly one ``bottle_gourd__<label>`` folder, and the loader
emits a multi-hot vector indexing into the global label vocabulary. The
Phase 11 full-OLID download will expand the label vocabulary; the
loader code does not need to change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.data_splits import load_class_map, load_split
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_DIR, PROJECT_ROOT

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)

OLID_DEFAULT_ROOT: Path = DATA_DIR / "causation" / "olid_i" / "raw"

_SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
_CONFIGS_DATA = PROJECT_ROOT / "configs" / "data"


def _labels_for(entry_label: str) -> list[str]:
    """Decompose an OLID I class-folder name into its multi-label tags.

    OLID I folder names use ``<crop>__<symptom>`` (e.g.
    ``bottle_gourd__DM``). Several folders carry **compound symptoms**
    where multiple co-occurring tags are joined with a single
    underscore (e.g. ``bottle_gourd__JAS_MIT`` = Jassid + Mite,
    ``ash_gourd__N_K`` = Nitrogen + Potassium deficiency,
    ``bitter_gourd__K_Mg`` = Potassium + Magnesium deficiency).

    Each of those constituent tags must light up its own bit in the
    multi-hot vector, otherwise the C5 evaluation can't separate the
    co-occurring conditions. We split:

    1. on ``__`` first to peel crop from the symptom block, and
    2. on ``_`` second to split a compound symptom block into individual
       tags.

    Examples
    --------
    >>> _labels_for("bottle_gourd__DM")
    ['bottle_gourd', 'DM']
    >>> _labels_for("bottle_gourd__JAS_MIT")
    ['bottle_gourd', 'JAS', 'MIT']
    >>> _labels_for("ash_gourd__N_K")
    ['ash_gourd', 'N', 'K']
    >>> _labels_for("tomato__healthy")
    ['tomato', 'healthy']
    >>> _labels_for("Alluvial_Soil")  # non-OLID fallback — unchanged
    ['Alluvial_Soil']
    """
    if "__" not in entry_label:
        return [entry_label]
    crop, symptoms = entry_label.split("__", 1)
    # ``symptoms`` may itself carry compound tags joined by a single ``_``.
    return [crop, *symptoms.split("_")]


class MultiLabelImageDataset:
    """Reads a Phase 4 split JSON and returns ``(image, multi_hot_vec)``.

    Parameters
    ----------
    split_path : Path
    raw_root : Path
    label_vocab : list[str]
        The ordered global label vocabulary. ``multi_hot_vec[i] = 1``
        means the image carries label ``label_vocab[i]``.
    transform : albumentations.Compose | None
    """

    def __init__(
        self,
        split_path: Path,
        raw_root: Path,
        label_vocab: list[str],
        transform: Any | None = None,
    ) -> None:
        self.entries = load_split(split_path)
        self.raw_root = Path(raw_root)
        self.label_vocab = list(label_vocab)
        self.label_to_idx = {lab: i for i, lab in enumerate(self.label_vocab)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        entry = self.entries[idx]
        image_path = self.raw_root / entry.path
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]

        multi_hot = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for label in _labels_for(entry.label):
            if label in self.label_to_idx:
                multi_hot[self.label_to_idx[label]] = 1.0
        return arr, multi_hot


def make_olid_loaders(
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, Any]:
    """train/val/test loaders for the OLID I smoke sample."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.integration.causation_transforms import (  # noqa: PLC0415
        build_olid_eval_aug,
        build_olid_train_aug,
    )
    from src.utils.data_stats import load_channel_stats  # noqa: PLC0415

    stats = load_channel_stats(_CONFIGS_DATA / "olid_i_norm.yaml")
    train_aug = build_olid_train_aug(380, stats.mean, stats.std)
    eval_aug = build_olid_eval_aug(380, stats.mean, stats.std)

    splits_dir = _SPLITS_ROOT / "olid_i"
    class_map = load_class_map(splits_dir / "class_map.json")
    label_vocab = sorted(class_map, key=lambda k: class_map[k])

    loaders: dict[str, DataLoader] = {}
    for split_name, transform, shuffle in (
        ("train", train_aug, True),
        ("val", eval_aug, False),
        ("test", eval_aug, False),
    ):
        ds = MultiLabelImageDataset(
            splits_dir / f"{split_name}.json",
            OLID_DEFAULT_ROOT,
            label_vocab=label_vocab,
            transform=transform,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    _LOGGER.info(
        "Built OLID loaders (smoke sample): train=%d, val=%d, test=%d, "
        "label_vocab=%d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
        len(label_vocab),
    )
    return loaders


__all__ = [
    "MultiLabelImageDataset",
    "OLID_DEFAULT_ROOT",
    "make_olid_loaders",
]
