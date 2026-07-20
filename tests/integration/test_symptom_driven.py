"""Symptom-driven retrieval guards.

The classical texts are general and SYMPTOM-based, not crop-specific (verified
against the literature and confirmed by the agronomy expert). Two consequences
are locked in here:

1. Strategy B must build a SYMPTOM-led query — the crop is background context,
   never the thing being searched for.
2. The grounded generator must not refuse merely because the retrieved passage
   does not name the user's crop; it refuses only when no passage addresses the
   observed condition.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.integration.causation import CausalContext, CausalPathway
from src.integration.config import LLMMediatedStrategyConfig
from src.integration.context import MultimodalContext
from src.integration.strategy_llm_mediated import LLMMediatedStrategy
from src.rag.generator import SYSTEM_PROMPT_V17


def _ctx(disease: str = "Potato leaf late blight", crop: str = "potato") -> MultimodalContext:
    return MultimodalContext(
        disease_pred=SimpleNamespace(class_name=disease, confidence=0.93),
        soil_pred=SimpleNamespace(soil_type="Alluvial_Soil",
                                  moisture_appearance="moderate", texture="mixed"),
        crop_type=crop,
        causal_context=CausalContext(pathway=CausalPathway.UNKNOWN, notes=None),
    )


def _prompt(ctx: MultimodalContext) -> str:
    return LLMMediatedStrategy(LLMMediatedStrategyConfig())._build_prompt(ctx)


# ------------------------------------------------------------------ #
# 1. Strategy B: symptom-led, crop demoted
# ------------------------------------------------------------------ #


def test_strategy_b_states_texts_are_symptom_based() -> None:
    p = _prompt(_ctx())
    assert "SYMPTOM-BASED" in p
    assert "not by crop species" in p or "not crop-specific" in p


def test_strategy_b_instructs_lead_with_symptom() -> None:
    p = _prompt(_ctx())
    assert "LEAD WITH THE SYMPTOM" in p
    # the crop-led anti-patterns must be explicitly warned against
    assert "retrieve poorly" in p


def test_strategy_b_demotes_crop_to_background_context() -> None:
    p = _prompt(_ctx())
    assert "background context only" in p
    # the crop value is still supplied (as context), just not as the search key
    assert "potato" in p


def test_strategy_b_still_carries_the_detected_disease() -> None:
    """Symptom-first must not drop the disease label — it's what we translate."""
    p = _prompt(_ctx(disease="Corn rust leaf", crop="corn"))
    assert "Corn rust leaf" in p


# ------------------------------------------------------------------ #
# 2. Generator: crop-agnostic grounding (no over-refusal)
# ------------------------------------------------------------------ #


def test_generator_keeps_the_refusal_guardrail() -> None:
    """Faithfulness is NOT weakened — refusal still exists for no evidence."""
    assert "ANSWER ONLY FROM THE RETRIEVED PASSAGES" in SYSTEM_PROMPT_V17
    assert "do not contain enough" in SYSTEM_PROMPT_V17


def test_generator_scopes_refusal_so_missing_crop_is_not_a_refusal() -> None:
    p = SYSTEM_PROMPT_V17
    assert "1a." in p, "rule 1a (scope of refusal) must be present"
    assert "SYMPTOM-BASED, NOT" in p
    assert "crop is absent" in p          # explicit: absence of crop != insufficient
    assert "no retrieved passage addresses the condition" in p


def test_generator_requires_flagging_general_passages() -> None:
    """When leaning on a general passage, the answer must say so honestly."""
    assert "generally" in SYSTEM_PROMPT_V17
