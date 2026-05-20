"""Smoke tests for the integration module."""

from __future__ import annotations

from src.integration import (
    CausalContext,
    CausalPathway,
    IntegrationConfig,
    LLMMediatedStrategy,
    MultimodalContext,
    MultimodalEmbeddingStrategy,
    TemplateStrategy,
)


def test_integration_config_defaults() -> None:
    cfg = IntegrationConfig()
    assert cfg.strategy == "template"
    assert cfg.require_causal_context is True


def test_causal_pathway_values() -> None:
    assert CausalPathway.SOIL_DRIVEN.value == "soil_driven"
    assert CausalPathway.PEST_VECTOR.value == "pest_vector"
    assert CausalPathway.CONTAGION.value == "contagion"
    assert CausalPathway.UNKNOWN.value == "unknown"


def test_causal_context_is_user_provided_per_c5() -> None:
    """C5: pathway comes from the user, not from an image-inference call."""
    ctx = CausalContext(pathway=CausalPathway.SOIL_DRIVEN, notes="Field waterlogged.")
    assert ctx.pathway is CausalPathway.SOIL_DRIVEN
    # Frozen — mutating must fail.
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.pathway = CausalPathway.UNKNOWN  # type: ignore[misc]


def test_strategy_classes_instantiate() -> None:
    cfg = IntegrationConfig()
    TemplateStrategy(cfg.template)
    LLMMediatedStrategy(cfg.llm_mediated)
    MultimodalEmbeddingStrategy(cfg.multimodal_embedding)


def test_multimodal_context_docstring_mentions_no_image_causation() -> None:
    """Module-level / class-level docstring must mention C5."""
    from src.integration import context as ctx_mod

    combined = (ctx_mod.__doc__ or "") + (MultimodalContext.__doc__ or "")
    assert "NOT infer" in combined or "user-provided" in combined.lower()
