"""Pydantic schema for the plant disease module.

Per master reference §10 (Disease Module): backbone is EfficientNet-B4
with 380×380 inputs; per the Phase 5 locked stack, training is a
three-stage cascade — PlantVillage pretraining → Paddy Doctor
fine-tune → PlantDoc fine-tune — with mixed-precision, gradient
clipping, and a brief frozen-backbone warm-up at the start of each
stage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.utils.config import BaseConfig


class AugmentationConfig(BaseConfig):
    """Image augmentation hyperparameters (albumentations-compatible).

    Kept deliberately small. The full pipeline is built in
    :mod:`src.disease.transforms` and reads these values to instantiate
    transforms.
    """

    horizontal_flip_p: float = Field(0.5, ge=0.0, le=1.0)
    vertical_flip_p: float = Field(0.0, ge=0.0, le=1.0)
    rotation_limit: int = Field(15, ge=0, le=180)
    brightness_limit: float = Field(0.2, ge=0.0, le=1.0)
    contrast_limit: float = Field(0.2, ge=0.0, le=1.0)
    # Default ImageNet mean/std; the live training loop reads the
    # per-dataset values from configs/data/<name>_norm.yaml.
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class DiseaseConfig(BaseConfig):
    """Configuration for the plant disease classifier (Phase 5 §D)."""

    # ----- locked stack -------------------------------------------------- #
    backbone: Literal["efficientnet_b4"] = "efficientnet_b4"
    image_size: int = Field(380, ge=64, description="EfficientNet-B4 native input is 380.")
    pretrained: bool = True
    dropout_rate: float = Field(0.3, ge=0.0, le=0.8)

    # ----- stage epoch budgets (Phase 5 prompt decision #11) ------------ #
    pretrain_epochs: int = Field(25, ge=1)
    finetune_paddy_epochs: int = Field(20, ge=1)
    finetune_plantdoc_epochs: int = Field(30, ge=1)
    freeze_backbone_epochs: int = Field(
        3,
        ge=0,
        description=(
            "Epochs to keep the backbone frozen at the start of each "
            "training stage (head warm-up)."
        ),
    )

    # ----- optimisation ------------------------------------------------- #
    lr_head: float = Field(1e-3, gt=0.0, description="LR for the classifier head.")
    lr_backbone: float = Field(
        1e-4,
        gt=0.0,
        description="LR for the backbone once unfrozen (typically 10x lower than the head).",
    )
    weight_decay: float = Field(1e-4, ge=0.0)
    gradient_clip: float = Field(1.0, ge=0.0, description="L2-norm clip; 0 disables.")
    mixed_precision: bool = Field(
        True,
        description="Enable torch.cuda.amp on CUDA devices. CPU falls back to fp32.",
    )

    # ----- batch / dataloader ------------------------------------------- #
    batch_size: int = Field(
        16,
        ge=1,
        description=(
            "Default Colab-T4-safe batch size. The train script's "
            "auto_batch_size() may override based on detected VRAM."
        ),
    )
    num_workers: int = Field(2, ge=0)

    # ----- reproducibility ---------------------------------------------- #
    seed: int = Field(42, ge=0)

    # ----- legacy aliases (kept for backward compatibility) -------------- #
    # Older tests / config files reference `num_classes` / `epochs` / `lr`.
    # Keep them as optional fields the training loop ignores; the real
    # num_classes is set per dataset by the DataLoader's class_map.
    num_classes: int | None = Field(
        default=None,
        description=(
            "Optional legacy field. The training loop derives num_classes "
            "per stage from each dataset's class_map.json."
        ),
    )

    # Per-stage augmentation hyperparameters (transforms.py uses these).
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
