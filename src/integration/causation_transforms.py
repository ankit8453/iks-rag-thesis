"""Augmentation pipelines for OLID I (causation evaluation, C5).

**Colour augmentation is deliberately dropped** for OLID. Several of
the labels in the multi-label set correspond to nutrient deficiencies
whose visual cue is hue (yellowing, browning, dark spots). Augmenting
brightness / contrast / saturation / hue would either erase the cue
(false negatives at train time) or invent it (false positives). Only
**geometric** augmentation is kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albumentations import Compose


def build_olid_train_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Geometric-only train augmentation for OLID I."""
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_olid_eval_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Deterministic eval pipeline."""
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    return A.Compose(
        [
            A.Resize(image_size + 32, image_size + 32),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


__all__ = ["build_olid_eval_aug", "build_olid_train_aug"]
