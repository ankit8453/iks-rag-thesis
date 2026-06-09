"""Phase 8 — Strategy A (Template) tests.

Deterministic, no models, no network. Validates:

- Every structured field (disease, crop, soil_type, moisture, texture)
  appears in the rendered query.
- The causal-pathway clause is present iff a non-``UNKNOWN`` pathway is
  supplied.
- ``UNKNOWN`` yields NO causal clause (matches the "default unspecified"
  contract from the Phase 8 prompt).
- The same context renders the same string twice (determinism).
- The humanisation step strips dataset noise (underscores, trailing
  ``_Soil``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.integration.causation import CausalContext, CausalPathway
from src.integration.config import TemplateStrategyConfig
from src.integration.context import MultimodalContext
from src.integration.strategy_template import TemplateStrategy


# --------------------------------------------------------------------- #
# Fixtures: minimal duck-typed disease + soil predictions. We do NOT
# import DiseasePrediction / SoilPrediction concretely because that
# would pull pydantic / pytorch into a pure-python unit test.
# --------------------------------------------------------------------- #


@dataclass
class _StubDiseasePred:
    class_name: str
    confidence: float = 0.92
    class_index: int = 0
    logits: list[float] | None = None


@dataclass
class _StubSoilPred:
    soil_type: str
    moisture_appearance: str
    texture: str
    per_head_confidence: dict | None = None


def _make_ctx(
    disease: str = "Tomato___Leaf_Mold",
    crop: str = "tomato",
    soil_type: str = "Alluvial_Soil",
    moisture: str = "moderate",
    texture: str = "mixed",
    pathway: CausalPathway = CausalPathway.UNKNOWN,
) -> MultimodalContext:
    return MultimodalContext(
        disease_pred=_StubDiseasePred(class_name=disease),
        soil_pred=_StubSoilPred(
            soil_type=soil_type,
            moisture_appearance=moisture,
            texture=texture,
        ),
        crop_type=crop,
        causal_context=CausalContext(pathway=pathway),
    )


@pytest.fixture()
def strategy() -> TemplateStrategy:
    return TemplateStrategy(TemplateStrategyConfig())


# --------------------------------------------------------------------- #
# Field-coverage tests
# --------------------------------------------------------------------- #


def test_template_includes_all_structured_fields(strategy: TemplateStrategy) -> None:
    ctx = _make_ctx(
        disease="Tomato___Leaf_Mold",
        crop="tomato",
        soil_type="Alluvial_Soil",
        moisture="moist",
        texture="mixed",
    )
    q = strategy.build_query(ctx)
    # Disease label humanised (no underscores).
    assert "tomato leaf mold" in q.lower()
    # Crop appears.
    assert "tomato" in q.lower()
    # Soil type humanised (no '_Soil' suffix, no underscores).
    assert "alluvial" in q.lower()
    assert "alluvial soil soil" not in q.lower()  # double-soil bug
    # Moisture appears.
    assert "moist" in q.lower()
    # Texture appears.
    assert "mixed" in q.lower()


def test_template_strips_dataset_label_noise(strategy: TemplateStrategy) -> None:
    """``__`` / ``_`` separators and trailing ``_Soil`` should be gone."""
    ctx = _make_ctx(
        disease="Rice___Blast_Disease",
        soil_type="Black_Soil",
    )
    q = strategy.build_query(ctx)
    assert "___" not in q
    assert "_Soil" not in q
    assert "Black_Soil" not in q
    assert "black" in q.lower()


# --------------------------------------------------------------------- #
# Causal-pathway clause matrix
# --------------------------------------------------------------------- #


def test_template_no_causal_clause_when_unknown(strategy: TemplateStrategy) -> None:
    """``UNKNOWN`` MUST suppress the pathway clause — Phase 8 prompt
    locks ``"unspecified" / UNKNOWN`` as the default and no-bias case."""
    q = strategy.build_query(_make_ctx(pathway=CausalPathway.UNKNOWN))
    # No clause keywords should appear.
    for keyword in (
        "emphasis on soil",
        "emphasis on pest",
        "preventing the spread",
    ):
        assert keyword not in q, f"unknown pathway leaked clause: {keyword!r}"


@pytest.mark.parametrize(
    "pathway, expected_keyword",
    [
        (CausalPathway.SOIL_DRIVEN, "soil restoration"),
        (CausalPathway.PEST_VECTOR, "pest"),
        (CausalPathway.CONTAGION, "spread"),
    ],
)
def test_template_appends_pathway_clause_when_set(
    strategy: TemplateStrategy,
    pathway: CausalPathway,
    expected_keyword: str,
) -> None:
    q = strategy.build_query(_make_ctx(pathway=pathway))
    assert expected_keyword in q.lower(), (
        f"pathway={pathway} should append a clause containing {expected_keyword!r}; "
        f"got: {q!r}"
    )


# --------------------------------------------------------------------- #
# Determinism + edge cases
# --------------------------------------------------------------------- #


def test_template_is_deterministic(strategy: TemplateStrategy) -> None:
    ctx = _make_ctx()
    assert strategy.build_query(ctx) == strategy.build_query(ctx)


def test_template_handles_empty_crop(strategy: TemplateStrategy) -> None:
    """An empty crop string should not crash — it should fall back to a
    generic phrase so the query is still well-formed."""
    ctx = _make_ctx(crop="")
    q = strategy.build_query(ctx)
    assert "crops" in q.lower() or "crop" in q.lower()
