"""Grad-CAM explanations for disease predictions (Phase 6).

Wraps :mod:`pytorch_grad_cam` so the disease classifier exposes its
attention map for any prediction. The integration tests will verify the
heatmap shape matches the input image; the qualitative review with the
supervisor happens in Phase 6 / Week 20.
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
) -> None:
    """Compute and save a Grad-CAM overlay for a single image.

    Parameters
    ----------
    classifier : DiseaseClassifier
        A trained classifier with weights loaded.
    image_path : Path
        Path to the input image.
    output_path : Path
        Where to write the PNG overlay.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 6 — Week 20.
    """
    raise NotImplementedError("Phase 6 — Week 20: implement Grad-CAM overlay.")
