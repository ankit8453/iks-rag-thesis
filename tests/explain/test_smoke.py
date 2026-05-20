"""Smoke tests for the explain module."""

from __future__ import annotations

from src.explain import compute_gradcam, highlight_chunks
from src.explain.chunk_highlight import HighlightedChunk, HighlightedSpan


def test_explain_classes_have_docstrings() -> None:
    assert compute_gradcam.__doc__ is not None
    assert highlight_chunks.__doc__ is not None
    assert HighlightedChunk.__doc__ is not None
    assert HighlightedSpan.__doc__ is not None
