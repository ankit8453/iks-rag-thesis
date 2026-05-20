"""Dataset wrappers for plant disease training and evaluation.

Implementation lives in Phase 5 (Week 16). Stubs here document the
expected file layout so the data engineering step is unambiguous.

Per master reference §41 the image datasets live under the top-level
``data/`` directory, not inside ``corpus/`` (which is reserved for the
IKS classical-text corpus). The defaults below resolve from
:data:`src.utils.paths.DATA_PLANT_DISEASE_DIR`.

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
from src.utils.paths import DATA_PLANT_DISEASE_DIR

if TYPE_CHECKING:
    from torch.utils.data import Dataset  # noqa: F401

PLANTVILLAGE_DEFAULT_ROOT: Path = DATA_PLANT_DISEASE_DIR / "PlantVillage"
PLANTDOC_DEFAULT_ROOT: Path = DATA_PLANT_DISEASE_DIR / "PlantDoc"


class PlantVillageDataset:
    """PlantVillage dataset wrapper (38-class, ~54k images).

    Parameters
    ----------
    root : Path
        Path to the unpacked PlantVillage directory. Defaults to
        :data:`PLANTVILLAGE_DEFAULT_ROOT`
        (``<repo>/data/plant_disease/PlantVillage``).
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

    def __init__(
        self,
        config: DiseaseConfig,
        root: Path | None = None,
        split: str = "train",
    ) -> None:
        self.root = Path(root) if root is not None else PLANTVILLAGE_DEFAULT_ROOT
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

    Parameters
    ----------
    config : DiseaseConfig
    root : Path, optional
        Defaults to :data:`PLANTDOC_DEFAULT_ROOT`
        (``<repo>/data/plant_disease/PlantDoc``).
    """

    def __init__(self, config: DiseaseConfig, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else PLANTDOC_DEFAULT_ROOT
        self.config = config

    def __len__(self) -> int:
        raise NotImplementedError("Phase 5 — Week 18: implement PlantDocDataset.__len__.")

    def __getitem__(self, idx: int) -> tuple[object, int]:
        raise NotImplementedError("Phase 5 — Week 18: implement PlantDocDataset.__getitem__.")
