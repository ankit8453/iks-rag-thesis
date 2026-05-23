"""Plant disease classification module.

Implements contribution C1 on the disease side: an EfficientNet-B4
classifier fine-tuned on PlantVillage (38 classes). Phase 4 adds the
dataset / split / DataLoader plumbing; the actual model training lands
in Phase 5 (Weeks 16-19).
"""

from src.disease.config import AugmentationConfig, DiseaseConfig
from src.disease.dataset import (
    JSONIndexedImageDataset,
    PlantDocDataset,
    PlantVillageDataset,
    make_paddy_doctor_loaders,
    make_plantdoc_loaders,
    make_plantvillage_loaders,
)
from src.disease.model import DiseaseClassifier, DiseasePrediction

__all__ = [
    "AugmentationConfig",
    "DiseaseClassifier",
    "DiseaseConfig",
    "DiseasePrediction",
    "JSONIndexedImageDataset",
    "PlantDocDataset",
    "PlantVillageDataset",
    "make_paddy_doctor_loaders",
    "make_plantdoc_loaders",
    "make_plantvillage_loaders",
]
