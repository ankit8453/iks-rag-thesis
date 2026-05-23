"""Albumentations pipelines for the soil module.

Soil augmentation is **heavier** than disease augmentation because:

- The Phantom-fs dataset has ~1,200 training images across 7 classes,
  vs PlantVillage's ~43,000 across 38 classes. EfficientNet-B0 (or
  any modern CNN) overfits on the small set without strong regularisation.
- Soil textures look broadly similar across rotations and flips —
  horizontal and vertical flips both add training signal without
  changing class identity.
- Random erasing / CoarseDropout simulates partial occlusion which is
  common in field-collected soil photographs (tools, vegetation cover).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albumentations import Compose


def build_soil_train_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Heavier train-time pipeline for the small soil dataset."""
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.85, 1.15)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_soil_eval_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Deterministic eval pipeline: resize + center crop."""
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


__all__ = ["build_soil_eval_aug", "build_soil_train_aug"]
