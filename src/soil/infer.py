"""Single-image inference for the soil classifier (Phase 6)."""

from __future__ import annotations

from pathlib import Path

from src.soil.config import SoilConfig
from src.soil.model import SoilPrediction


def predict_image(image_path: Path, checkpoint_path: Path, config: SoilConfig) -> SoilPrediction:
    """Predict all visual soil attributes for one image.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 6 — Week 19.
    """
    raise NotImplementedError("Phase 6 — Week 19: implement single-image soil inference.")
