"""Albumentations pipelines for the soil module (Phase 6 §B).

Phase 6 train pipeline matches the spec in
``PHASE6_PROMPT2_COLAB_NOTEBOOK.md``: a deliberately modest set of
augmentations. The combined soil corpus (~2,600 images across three
heterogeneous sources) gets enough natural variance from the per-source
photo conditions; heavy synthetic augmentation hurt accuracy in
earlier Phase 4 cross-region experiments.

The eval pipeline upscales to 256 then center-crops to 224, mirroring
the Phase 5 disease eval pattern (resize-then-crop preserves a small
border margin around the subject).
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
    """Train-time augmentation pipeline for Phase 6 soil training."""
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.8, 1.0),
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_soil_eval_aug(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """Deterministic eval pipeline: resize to 256, center-crop to ``image_size``."""
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    return A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


__all__ = ["build_soil_eval_aug", "build_soil_train_aug"]
