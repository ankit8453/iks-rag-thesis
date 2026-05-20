"""Pydantic schema for the plant disease module.

Per master reference §10 (Disease Module): backbone is EfficientNet-B4 with
~50k PlantVillage images; 38 classes (PlantVillage taxonomy).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.utils.config import BaseConfig


class AugmentationConfig(BaseConfig):
    """Image augmentation hyperparameters (albumentations-compatible).

    Kept deliberately small. The full pipeline is built in
    ``src/disease/dataset.py`` and reads these values to instantiate
    transforms.
    """

    horizontal_flip_p: float = Field(0.5, ge=0.0, le=1.0)
    vertical_flip_p: float = Field(0.0, ge=0.0, le=1.0)
    rotation_limit: int = Field(15, ge=0, le=180)
    brightness_limit: float = Field(0.2, ge=0.0, le=1.0)
    contrast_limit: float = Field(0.2, ge=0.0, le=1.0)
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class DiseaseConfig(BaseConfig):
    """Configuration for the plant disease classifier."""

    backbone: Literal["efficientnet_b4"] = "efficientnet_b4"
    num_classes: int = Field(38, ge=2, description="PlantVillage has 38 classes.")
    image_size: int = Field(380, ge=64, description="EfficientNet-B4 native input is 380.")
    batch_size: int = Field(32, ge=1)
    lr: float = Field(1e-4, gt=0.0)
    epochs: int = Field(30, ge=1)
    freeze_backbone_epochs: int = Field(
        5,
        ge=0,
        description="Number of epochs to keep the backbone frozen at the start of training.",
    )
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    seed: int = Field(42, ge=0)
    pretrained: bool = True
