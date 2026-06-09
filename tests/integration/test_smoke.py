"""Phase 8 integration-package smoke tests.

Pure import + dataclass shape checks. No GPU, no network, no model load.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.integration import (
    CausalContext,
    CausalPathway,
    IntegrationConfig,
    LLMMediatedStrategy,
    MultimodalContext,
    MultimodalEmbeddingStrategy,
    MultimodalProjector,
    TemplateStrategy,
    qualitative_compare,
    run_all_strategies,
)
from src.integration import context as ctx_mod


def test_integration_config_defaults() -> None:
    cfg = IntegrationConfig()
    assert cfg.strategy == "template"
    assert cfg.require_causal_context is True


def test_causal_pathway_values() -> None:
    """C5 enum locked: 4 values, ``UNKNOWN`` is the default-safe one."""
    assert CausalPathway.SOIL_DRIVEN.value == "soil_driven"
    assert CausalPathway.PEST_VECTOR.value == "pest_vector"
    assert CausalPathway.CONTAGION.value == "contagion"
    assert CausalPathway.UNKNOWN.value == "unknown"


def test_causal_context_is_user_provided_per_c5() -> None:
    """C5: pathway comes from the user, not from an image-inference call."""
    ctx = CausalContext(pathway=CausalPathway.SOIL_DRIVEN, notes="Field waterlogged.")
    assert ctx.pathway is CausalPathway.SOIL_DRIVEN
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.pathway = CausalPathway.UNKNOWN  # type: ignore[misc]


def test_strategy_classes_instantiate() -> None:
    cfg = IntegrationConfig()
    TemplateStrategy(cfg.template)
    LLMMediatedStrategy(cfg.llm_mediated)
    MultimodalEmbeddingStrategy(cfg.multimodal_embedding)


def test_multimodal_projector_shape() -> None:
    """Strategy C linear maps (disease + soil + crop) → 1024 (corpus dim)."""
    import torch

    p = MultimodalProjector(disease_dim=1792, soil_dim=1280, crop_dim=1024, out_dim=1024)
    x = torch.zeros((2, 1792 + 1280 + 1024))
    y = p(x)
    assert y.shape == (2, 1024)


def test_multimodal_context_docstring_mentions_no_image_causation() -> None:
    """Insurance against accidental scope creep — the module + class
    docstring together must still state that causal pathway comes from
    the user, not the image."""
    combined = (ctx_mod.__doc__ or "") + (MultimodalContext.__doc__ or "")
    assert "NOT infer" in combined or "user-supplied" in combined.lower() or "user-provided" in combined.lower()


def test_run_all_strategies_and_qualitative_compare_importable() -> None:
    """These two are the public API surface for Phase 8 notebooks."""
    assert callable(run_all_strategies)
    assert callable(qualitative_compare)
