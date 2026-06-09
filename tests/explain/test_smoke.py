"""Phase 9 explain-package smoke tests.

Pure import + docstring + back-compat shape checks. No GPU, no network.
"""

from __future__ import annotations

from src.explain import (
    ExplainedChunk,
    GradCAMResult,
    HighlightedChunk,
    HighlightedSpan,
    SOIL_HEADS,
    SoilHeadWrapper,
    compute_gradcam,
    disease_gradcam,
    explain_chunks,
    find_target_layer,
    highlight_chunks,
    render_retrieval_panel,
    render_vision_panel,
    save_explanation,
    soil_gradcam,
    tokenize,
)


def test_phase9_public_symbols_have_docstrings() -> None:
    for name in (
        disease_gradcam, soil_gradcam, find_target_layer, SoilHeadWrapper,
        explain_chunks, tokenize,
        render_vision_panel, render_retrieval_panel, save_explanation,
    ):
        assert name.__doc__ is not None and name.__doc__.strip(), (
            f"{name.__name__}: missing or empty docstring"
        )
    for cls in (GradCAMResult, ExplainedChunk):
        assert cls.__doc__ is not None and cls.__doc__.strip(), cls.__name__


def test_soil_heads_locked_to_three() -> None:
    """Soil model has three visual heads; SoilHeadWrapper must accept
    exactly those three names."""
    assert SOIL_HEADS == ("soil_type", "moisture", "texture")


def test_backcompat_shims_still_callable() -> None:
    """The earlier scaffold's ``compute_gradcam`` and
    ``highlight_chunks`` are kept so callers that only depend on
    their import-shape still work."""
    assert callable(compute_gradcam)
    assert callable(highlight_chunks)
    # highlight_chunks shim returns one HighlightedChunk per input,
    # with empty spans (Phase 9 explains retrieval, not generation).
    class _C:
        chunk_id = "c"
        text = "hello"
        score = 0.5
        metadata = {"source_text": "Foo"}
    out = highlight_chunks("any answer", [_C()])
    assert isinstance(out[0], HighlightedChunk)
    assert out[0].spans == []
    # HighlightedSpan exists too (Phase 10 UI consumes it).
    assert HighlightedSpan is not None
