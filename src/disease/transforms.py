"""Albumentations augmentation pipelines for the plant disease module.

Two factories — :func:`build_disease_train_aug` and
:func:`build_disease_eval_aug`. Each returns an
``albumentations.Compose`` object that takes a numpy ``HWC uint8`` image
and yields a normalised ``CHW float32`` torch tensor (via
``ToTensorV2``).

The numbers below (geometric limits, colour-jitter amplitudes) are
deliberately modest. PlantVillage / PlantDoc are already diverse enough
that heavy augmentation hurts; the soil module gets the heavier set
(see :mod:`src.soil.transforms`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albumentations import Compose


def build_disease_train_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Training-time augmentation pipeline for disease classifiers."""
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_disease_eval_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Eval-time augmentation pipeline (deterministic: resize + center crop)."""
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


__all__ = ["build_disease_eval_aug", "build_disease_train_aug"]
