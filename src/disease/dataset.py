"""Dataset wrappers for plant disease training and evaluation.

Implementation lives in Phase 5 (Week 16). Stubs here document the
expected file layout so the data engineering step is unambiguous.

Expected layout::

    data/plant_disease/PlantVillage/
        Apple___Apple_scab/*.jpg
        Apple___Black_rot/*.jpg
        ...

PlantDoc (out-of-distribution test set) lives under
``data/plant_disease/PlantDoc/`` with the same nested structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.disease.config import DiseaseConfig

if TYPE_CHECKING:
    from torch.utils.data import Dataset  # noqa: F401


class PlantVillageDataset:
    """PlantVillage dataset wrapper (38-class, ~54k images).

    Parameters
    ----------
    root : Path
        Path to the unpacked PlantVillage directory.
    config : DiseaseConfig
        Used for image size + augmentation.
    split : {"train", "val", "test"}
        Which split to load.

    Notes
    -----
    Phase 5 implementation will:
    - read class folders to build the label map
    - apply albumentations transforms per ``config.augmentation``
    - return ``(image_tensor, label)`` tuples
    """

    def __init__(self, root: Path, config: DiseaseConfig, split: str = "train") -> None:
        self.root = Path(root)
        self.config = config
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError("Phase 5 — Week 16: implement PlantVillageDataset.__len__.")

    def __getitem__(self, idx: int) -> tuple[object, int]:
        raise NotImplementedError("Phase 5 — Week 16: implement PlantVillageDataset.__getitem__.")


class PlantDocDataset:
    """PlantDoc out-of-distribution evaluation set.

    Same interface as :class:`PlantVillageDataset`. Used only at test time
    to measure OOD generalisation; never in the training mix.
    """

    def __init__(self, root: Path, config: DiseaseConfig) -> None:
        self.root = Path(root)
        self.config = config

    def __len__(self) -> int:
        raise NotImplementedError("Phase 5 — Week 18: implement PlantDocDataset.__len__.")

    def __getitem__(self, idx: int) -> tuple[object, int]:
        raise NotImplementedError("Phase 5 — Week 18: implement PlantDocDataset.__getitem__.")
