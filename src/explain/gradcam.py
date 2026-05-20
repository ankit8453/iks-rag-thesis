"""Grad-CAM wrapper using :mod:`pytorch_grad_cam`.

Used by both the disease module (Phase 6) and the demo UI (Phase 8). The
wrapper hides the choice of target-layer hook so callers only pass an
image and a classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.disease.model import DiseaseClassifier


def compute_gradcam(
    classifier: "DiseaseClassifier",
    image_path: Path,
    output_path: Path,
    target_class: int | None = None,
) -> Path:
    """Generate a Grad-CAM heatmap overlay for one image.

    Parameters
    ----------
    classifier : DiseaseClassifier
        A trained, loaded disease classifier.
    image_path : Path
        Path to the input image.
    output_path : Path
        Where to write the PNG overlay.
    target_class : int, optional
        Class index to explain. If None, uses the argmax prediction.

    Returns
    -------
    Path
        ``output_path``, for convenience.

    Raises
    ------
    NotImplementedError
        Phase 6 — Week 20.
    """
    raise NotImplementedError("Phase 6 — Week 20: implement Grad-CAM wrapper.")
