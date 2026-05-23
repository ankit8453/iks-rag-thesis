"""Multi-task visual soil analysis module.

VISUAL ONLY. Per master reference §11 and supervisor guardrail #2, this
module is forbidden from outputting NPK, pH, fertility, organic matter %,
or any other chemical / quantitative property. Future PRs that attempt
to add such outputs should fail review.

Heads (post-Phase-4 reconciliation): ``soil_type`` (Phantom-fs 7-class
Indian deposits), ``moisture_appearance`` (Sirajganj 2025 dry/moderate/
wet), and ``texture`` (IRSID with the §14 coarse/fine/mixed mapping).
``surface`` and ``cover`` from the original Week-2 design were dropped
during the soil-parameter coverage audit (supervisor sign-off received).
"""

from src.soil.config import SoilConfig
from src.soil.dataset import (
    SoilSample,
    SoilTypeDataset,
    make_irsid_loaders,
    make_phantomfs_loaders,
    make_sirajganj_moisture_loaders,
    make_soil_cross_region_loaders,
)
from src.soil.model import SoilClassifier, SoilPrediction

__all__ = [
    "SoilClassifier",
    "SoilConfig",
    "SoilPrediction",
    "SoilSample",
    "SoilTypeDataset",
    "make_irsid_loaders",
    "make_phantomfs_loaders",
    "make_sirajganj_moisture_loaders",
    "make_soil_cross_region_loaders",
]
