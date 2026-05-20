"""Tests for src.rag.prompts.render_prompt."""

from __future__ import annotations

from src.rag.prompts import render_prompt


def test_render_prompt_substitutes_query_and_chunks(
    sample_retrieved_chunks: list[dict[str, str]],
) -> None:
    rendered = render_prompt(
        query="My tomato plant has curled leaves.",
        retrieved_chunks=sample_retrieved_chunks,
        causal_context="Soil-driven; field was waterlogged last week.",
    )
    assert "curled leaves" in rendered
    assert "vrikshayurveda:1" in rendered
    assert "waterlogged" in rendered


def test_render_prompt_substitutes_none_for_missing_causal_context(
    sample_retrieved_chunks: list[dict[str, str]],
) -> None:
    rendered = render_prompt(
        query="test",
        retrieved_chunks=sample_retrieved_chunks,
        causal_context=None,
    )
    assert "None provided" in rendered
