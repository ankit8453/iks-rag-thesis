"""Inference helpers for the disease classifier (Phase 5).

This file wraps :class:`~src.disease.model.DiseaseClassifier` for one-shot
prediction from an image file path. The HTTP / Streamlit demo in
``demo/`` will route here.
"""

from __future__ import annotations

from pathlib import Path

from src.disease.config import DiseaseConfig
from src.disease.model import DiseasePrediction


def predict_image(
    image_path: Path, checkpoint_path: Path, config: DiseaseConfig
) -> DiseasePrediction:
    """Run disease inference on a single image file.

    Parameters
    ----------
    image_path : Path
        Path to an RGB image (JPEG/PNG).
    checkpoint_path : Path
        Path to a ``DiseaseClassifier.state_dict()`` checkpoint saved by
        :func:`src.disease.train.train`.
    config : DiseaseConfig
        Must match the config used at training time.

    Returns
    -------
    DiseasePrediction
        Predicted class, confidence, and logits.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 5 — Week 17.
    """
    raise NotImplementedError("Phase 5 — Week 17: implement single-image inference.")
