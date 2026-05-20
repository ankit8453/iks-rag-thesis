"""Multi-task visual soil analysis module.

VISUAL ONLY. Per master reference §11 and supervisor guardrail #2, this
module is forbidden from outputting NPK, pH, fertility, organic matter %,
or any other chemical / quantitative property. Future PRs that attempt to
add such outputs should fail review.

Heads: ``soil_type``, ``texture``, ``surface``, ``moisture``, ``cover``.
"""

from src.soil.config import SoilConfig
from src.soil.dataset import SoilTypeDataset
from src.soil.model import SoilClassifier, SoilPrediction

__all__ = [
    "SoilClassifier",
    "SoilConfig",
    "SoilPrediction",
    "SoilTypeDataset",
]
