"""Multi-task visual soil analysis module.

VISUAL ONLY. Per master reference §11 and supervisor guardrail #2, this
module is forbidden from outputting NPK, pH, fertility, organic matter %,
or any other chemical / quantitative property. Future PRs that attempt
to add such outputs should fail review.

Heads (post-Phase-4 reconciliation): ``soil_type`` (Phantom-fs 7-class
Indian deposits), ``moisture`` (Sirajganj 2025 wet/moist/dry), and
``texture`` (IRSID + VIT, USDA-collapsed). ``surface`` and ``cover``
from the original Week-2 design were dropped during the soil-parameter
coverage audit (supervisor sign-off received).

Phase 6 multi-task training (Colab notebook in
``notebooks/phase6_soil_training.ipynb``) consumes:

- :class:`SoilMultiTaskClassifier` — EfficientNet-B0 + 3 dropout-linear heads
- :func:`build_soil_train_aug` / :func:`build_soil_eval_aug` — albumentations
- :data:`TASK_WEIGHTS`, :func:`compute_multitask_loss` — weighted CE with ignore_index=-1
- :func:`train_one_epoch`, :func:`evaluate_per_task` — per-epoch driver + per-task eval
- :class:`SoilCheckpointManager` — HF Hub-backed checkpoint round-trip
- :func:`auto_batch_size` — Colab T4 / L4 / A100 batch-size picker
- :func:`build_multitask_labels` — single-head label dict for multi-task fusion
"""

from src.soil.config import SoilConfig
from src.soil.dataset import (
    SoilSample,
    SoilTypeDataset,
    build_multitask_labels,
    make_irsid_loaders,
    make_phantomfs_loaders,
    make_sirajganj_moisture_loaders,
    make_soil_cross_region_loaders,
)
from src.soil.model import (
    DEFAULT_BACKBONE,
    HEAD_NUM_CLASSES,
    SoilClassifier,
    SoilMultiTaskClassifier,
    SoilPrediction,
)
from src.soil.train import (
    DEFAULT_MODEL_REPO,
    SoilCheckpointManager,
    TASK_WEIGHTS,
    auto_batch_size,
    compute_multitask_loss,
    evaluate_per_task,
    train_one_epoch,
)
from src.soil.transforms import build_soil_eval_aug, build_soil_train_aug

__all__ = [
    "DEFAULT_BACKBONE",
    "DEFAULT_MODEL_REPO",
    "HEAD_NUM_CLASSES",
    "SoilCheckpointManager",
    "SoilClassifier",
    "SoilConfig",
    "SoilMultiTaskClassifier",
    "SoilPrediction",
    "SoilSample",
    "SoilTypeDataset",
    "TASK_WEIGHTS",
    "auto_batch_size",
    "build_multitask_labels",
    "build_soil_eval_aug",
    "build_soil_train_aug",
    "compute_multitask_loss",
    "evaluate_per_task",
    "make_irsid_loaders",
    "make_phantomfs_loaders",
    "make_sirajganj_moisture_loaders",
    "make_soil_cross_region_loaders",
    "train_one_epoch",
]
