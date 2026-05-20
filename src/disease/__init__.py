"""Plant disease classification module.

Implements contribution C1 of the thesis on the disease side: an
EfficientNet-B4 classifier fine-tuned on PlantVillage (38 classes). All
implementation arrives in Phase 5 (Weeks 16-19); this package currently
exposes only stubs with stable interfaces so downstream modules
(integration, eval) can be written against them.
"""

from src.disease.config import AugmentationConfig, DiseaseConfig
from src.disease.dataset import PlantDocDataset, PlantVillageDataset
from src.disease.model import DiseaseClassifier, DiseasePrediction

__all__ = [
    "AugmentationConfig",
    "DiseaseClassifier",
    "DiseaseConfig",
    "DiseasePrediction",
    "PlantDocDataset",
    "PlantVillageDataset",
]
