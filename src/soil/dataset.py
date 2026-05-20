"""Soil-type dataset wrapper.

Targets the Kaggle "Soil Type Image Classification" dataset
(https://www.kaggle.com/datasets/abdulqayyum/soil-types-image-classification).
Expected on disk at ``data/soil/SoilTypes/`` after manual download.

Per guardrail #4 this dataset must support a region-held-out split for
cross-region validation. Each sample carries an optional ``region`` field
which the split builder uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.soil.config import SoilConfig


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
    root : Path
        Path to the unpacked ``SoilTypes`` directory.
    config : SoilConfig
        Used for image size + augmentation.
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
        root: Path,
        config: SoilConfig,
        split: str = "train",
        held_out_regions: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.split = split
        self.held_out_regions = held_out_regions or []

    def __len__(self) -> int:
        raise NotImplementedError("Phase 6 — Week 19: implement SoilTypeDataset.__len__.")

    def __getitem__(self, idx: int) -> tuple[object, dict[str, int]]:
        raise NotImplementedError("Phase 6 — Week 19: implement SoilTypeDataset.__getitem__.")
