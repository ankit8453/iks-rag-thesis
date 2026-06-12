"""Cropped-PlantDoc (C-PD) retrain — the fix for the PlantDoc-stage damage.

EXPERIMENT_LOG §6c established: full fine-tuning on FULL PlantDoc images
taught the classifier to use the background, damaging the healthy
leaf-attention features built in the PlantVillage + Paddy stages. Three
inference-side / training-side fixes failed because none forces the model
to look at the leaf.

This module builds a **cropped** training set: every image is cut down to
its ground-truth leaf box (from PlantDoc's VOC detection annotations), so
there is NO background left to learn from — the model is forced to learn
the leaf. This is the "Cropped-PlantDoc (C-PD)" recipe (Singh et al.,
CoDS-COMAD 2020), which reaches ~70% with honest leaf attention.

The heavy training loop is reused UNCHANGED from
:func:`src.disease.train.train_one_stage`. This module only adds the
cropped torch ``Dataset`` + loader builders. The backbone is warm-started
from the **PlantVillage** stage (general, multi-crop, leaf-attentive),
NOT the rice-narrow Paddy stage that LP-FT mistakenly froze.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

from src.disease.detect_crop import (
    Box,
    build_normalized_class_map,
    crop_to_box,
    match_class_to_index,
    parse_voc_xml,
)
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

#: Where the C-PD retrain pushes (NEW repo — leaves originals untouched).
DEFAULT_CROP_REPO: str = "ankit-iiitdmj/iks-disease-plantdoc-crop"

#: Warm-start backbone — the leaf-attentive PlantVillage stage.
DEFAULT_BACKBONE_REPO: str = "ankit-iiitdmj/iks-disease-plantvillage"

_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _image_for_xml(xml_path: str) -> str | None:
    """Return the image file paired with a VOC XML (same basename), or None."""
    base = os.path.splitext(xml_path)[0]
    for ext in _IMAGE_EXTS:
        if os.path.exists(base + ext):
            return base + ext
    return None


def index_cropped_samples(
    split_dir: str | Path, class_map: dict[str, int],
) -> list[tuple[str, Box, int]]:
    """Scan a PlantDoc detection split for (image_path, box, label_idx) crops.

    One entry per ground-truth box whose class name maps into ``class_map``.
    Boxes with an unmatched class name, or images that can't be located, are
    skipped (counts logged) — never crash mid-build.

    Returns a flat list; each element becomes one cropped training sample.
    """
    norm_map = build_normalized_class_map(class_map)
    xmls = sorted(glob.glob(os.path.join(str(split_dir), "*.xml")))
    samples: list[tuple[str, Box, int]] = []
    n_unmatched = n_noimg = 0
    for xml in xmls:
        ipath = _image_for_xml(xml)
        if ipath is None:
            n_noimg += 1
            continue
        for box in parse_voc_xml(xml):
            label = match_class_to_index(box.name, norm_map)
            if label is None:
                n_unmatched += 1
                continue
            samples.append((ipath, box, label))
    _LOGGER.info(
        "Cropped index for %s: %d samples (%d unmatched-class boxes, "
        "%d images missing).", split_dir, len(samples), n_unmatched, n_noimg,
    )
    return samples


def _make_cropped_dataset_class() -> Any:
    """Build the torch ``Dataset`` class lazily (avoids a torch import at
    module load)."""
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset

    class _CroppedBoxDataset(Dataset):
        def __init__(
            self,
            samples: list[tuple[str, Box, int]],
            transform: Any,
            pad_frac: float = 0.10,
        ) -> None:
            self.samples = samples
            self.transform = transform
            self.pad_frac = pad_frac

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> tuple[Any, int]:
            path, box, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            crop = crop_to_box(img, box, pad_frac=self.pad_frac)
            arr = np.asarray(crop)
            tensor = self.transform(image=arr)["image"]
            return tensor, int(label)

    return _CroppedBoxDataset


def build_cropped_loaders(
    train_dir: str | Path,
    class_map: dict[str, int],
    *,
    batch_size: int = 16,
    val_frac: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    image_size: int = 380,
    pad_frac: float = 0.10,
) -> tuple[Any, Any, int]:
    """Build train + val DataLoaders of leaf crops from a detection split.

    ``train_dir`` is the PlantDoc detection TRAIN folder (images + VOC XML).
    A ``val_frac`` slice is held out for validation. Returns
    ``(train_loader, val_loader, n_train_samples)``.
    """
    import random

    import torch
    from torch.utils.data import DataLoader

    from src.disease.transforms import (
        build_disease_eval_aug,
        build_disease_train_aug,
    )

    samples = index_cropped_samples(train_dir, class_map)
    if not samples:
        raise RuntimeError(
            f"No cropped samples found under {train_dir!r}. Check the "
            "detection-repo path and that class names match class_map.json."
        )
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = build_disease_train_aug(image_size, mean, std)
    eval_aug = build_disease_eval_aug(image_size, mean, std)

    ds_cls = _make_cropped_dataset_class()
    train_loader = DataLoader(
        ds_cls(train_samples, train_aug, pad_frac=pad_frac),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ds_cls(val_samples, eval_aug, pad_frac=pad_frac),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, len(train_samples)


def build_cropped_test_loader(
    test_dir: str | Path,
    class_map: dict[str, int],
    *,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 380,
    pad_frac: float = 0.10,
) -> tuple[Any, int]:
    """Build a held-out test loader of leaf crops from the detection TEST split.

    Returns ``(test_loader, n_samples)``. This split is disjoint from TRAIN
    by the detection repo's own split, so it's a valid held-out set for the
    crop-retrained model.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.disease.transforms import build_disease_eval_aug

    samples = index_cropped_samples(test_dir, class_map)
    if not samples:
        raise RuntimeError(f"No cropped test samples under {test_dir!r}.")
    eval_aug = build_disease_eval_aug(
        image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    )
    ds_cls = _make_cropped_dataset_class()
    loader = DataLoader(
        ds_cls(samples, eval_aug, pad_frac=pad_frac),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(samples)


def build_crop_model(
    num_classes: int = 27,
    backbone_repo: str = DEFAULT_BACKBONE_REPO,
) -> Any:
    """Build a 27-class model warm-started from the PlantVillage backbone.

    Full fine-tuning is fine here (unlike LP-FT) because the cropped inputs
    have no background to distort the features toward.
    """
    from src.disease.train import CheckpointManager, _build_model_for_stage

    state = CheckpointManager(backbone_repo).try_load_latest()
    if state is None:
        raise RuntimeError(
            f"No checkpoint at {backbone_repo!r}. Need the PlantVillage "
            "backbone to warm-start the C-PD retrain."
        )
    model = _build_model_for_stage("finetune_plantdoc", num_classes, state)
    _LOGGER.info("C-PD model ready: PlantVillage backbone + fresh %d-class head.", num_classes)
    return model


__all__ = [
    "DEFAULT_BACKBONE_REPO",
    "DEFAULT_CROP_REPO",
    "build_crop_model",
    "build_cropped_loaders",
    "build_cropped_test_loader",
    "index_cropped_samples",
]
