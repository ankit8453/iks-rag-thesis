"""Explainability layer (Phase 9 §C).

Two surfaces:

- **Vision Grad-CAM** —
  :func:`~src.explain.gradcam.disease_gradcam` and
  :func:`~src.explain.gradcam.soil_gradcam` (the latter via
  :class:`~src.explain.gradcam.SoilHeadWrapper` so the multi-task soil
  model's three heads can each be attributed independently).
- **Retrieved-chunk highlighting** —
  :func:`~src.explain.chunk_highlight.explain_chunks` produces a
  per-chunk record with the query↔chunk matched terms and a
  ``text_with_markers`` rendering (the Streamlit UI / notebook
  converts the ``**…**`` markers to its own highlight syntax).

Matplotlib renderers (:func:`~src.explain.visualize.render_vision_panel`
+ :func:`~src.explain.visualize.render_retrieval_panel`) bundle the two
surfaces into the figures the Phase 10 UI surfaces and the paper
reuses.
"""

from src.explain.chunk_highlight import (
    ExplainedChunk,
    HighlightedChunk,
    HighlightedSpan,
    explain_chunks,
    highlight_chunks,
    tokenize,
)
from src.explain.gradcam import (
    SOIL_HEADS,
    GradCAMResult,
    SoilHeadWrapper,
    compute_gradcam,
    disease_gradcam,
    find_target_layer,
    soil_gradcam,
)
from src.explain.visualize import (
    DEFAULT_OUT_ROOT,
    render_retrieval_panel,
    render_vision_panel,
    save_explanation,
)

__all__ = [
    "DEFAULT_OUT_ROOT",
    "ExplainedChunk",
    "GradCAMResult",
    "HighlightedChunk",
    "HighlightedSpan",
    "SOIL_HEADS",
    "SoilHeadWrapper",
    "compute_gradcam",
    "disease_gradcam",
    "explain_chunks",
    "find_target_layer",
    "highlight_chunks",
    "render_retrieval_panel",
    "render_vision_panel",
    "save_explanation",
    "soil_gradcam",
    "tokenize",
]
