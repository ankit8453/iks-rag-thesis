"""Soil-type dataset wrapper.

Targets the Kaggle "Soil Type Image Classification" dataset
(https://www.kaggle.com/datasets/abdulqayyum/soil-types-image-classification).

Per master reference §41 the dataset lives under the top-level ``data/``
directory, never inside ``corpus/``. The default root resolves from
:data:`src.utils.paths.DATA_SOIL_DIR` to
``<repo>/data/soil/SoilTypes/``.

Per guardrail #4 this dataset must support a region-held-out split for
cross-region validation. Each sample carries an optional ``region`` field
which the split builder uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.soil.config import SoilConfig
from src.utils.paths import DATA_SOIL_DIR

SOIL_TYPES_DEFAULT_ROOT: Path = DATA_SOIL_DIR / "SoilTypes"


@dataclass
class SoilSample:
    """One soil image plus its multi-task labels and provenance.

    Attributes
    ----------
    image_path : Path
        Path to the source image.
    soil_type : str
        Categorical label: alluvial / black / clay / red / sandy / ...
    texture : str
        Visual texture label.
    surface : str
        Visual surface label (e.g. cracked, smooth).
    moisture_appearance : str
        Visual moisture label (e.g. wet, dry, moist).
    cover_state : str
        Cover label (e.g. bare, vegetated, debris).
    region : str | None
        Geographic region of capture, when available. Drives the
        region-held-out validation split.
    """

    image_path: Path
    soil_type: str
    texture: str
    surface: str
    moisture_appearance: str
    cover_state: str
    region: str | None = None


class SoilTypeDataset:
    """Soil dataset wrapper supporting random and region-held-out splits.

    Parameters
    ----------
    config : SoilConfig
        Used for image size + augmentation.
    root : Path, optional
        Defaults to :data:`SOIL_TYPES_DEFAULT_ROOT`
        (``<repo>/data/soil/SoilTypes``).
    split : {"train", "val", "test"}
        Which split to load.
    held_out_regions : list[str] | None
        If provided (and ``config.cross_region_validation`` is True),
        these regions are excluded from train and form the val/test sets.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 6 — Week 19.
    """

    def __init__(
        self,
        config: SoilConfig,
        root: Path | None = None,
        split: str = "train",
        held_out_regions: list[str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else SOIL_TYPES_DEFAULT_ROOT
        self.config = config
        self.split = split
        self.held_out_regions = held_out_regions or []

    def __len__(self) -> int:
        raise NotImplementedError("Phase 6 — Week 19: implement SoilTypeDataset.__len__.")

    def __getitem__(self, idx: int) -> tuple[object, dict[str, int]]:
        raise NotImplementedError("Phase 6 — Week 19: implement SoilTypeDataset.__getitem__.")
