"""Explainability utilities for vision and RAG.

- :func:`compute_gradcam` — Grad-CAM overlay for a disease prediction.
- :func:`highlight_chunks` — render retrieved chunks with span-level
  highlights aligned to a generated answer.
"""

from src.explain.chunk_highlight import HighlightedChunk, highlight_chunks
from src.explain.gradcam import compute_gradcam

__all__ = [
    "HighlightedChunk",
    "compute_gradcam",
    "highlight_chunks",
]
