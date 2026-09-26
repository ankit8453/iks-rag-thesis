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


# ------------------------------------------------------------------ #
# Bridge-quality rules added after §6o measured what Strategy B really writes.
#
# Stage 0 scored Llama's actual output against hand-written queries and lost
# 14-0 (mean top-1 0.0445 vs 0.1691, 16 of 17 below the 0.15 floor). Every
# generated query broke the prompt the same three ways: it opened with
# "In loam soil with moderate moisture", it ended by asking what was
# "described in the classical Sanskrit treatises", and it kept modern
# pathology words. These tests pin the instructions that address each fault,
# so the fix cannot be silently lost in a later edit.
# ------------------------------------------------------------------ #


def test_prompt_requires_a_description_not_a_question() -> None:
    """Llama asked the texts about themselves; the corpus is written in statements."""
    p = _prompt(_ctx())
    assert "NOT a question" in p
    assert "Never end with a question mark" in p


def test_prompt_forbids_mentioning_the_texts() -> None:
    """'...described in the classical Sanskrit treatises?' matched nothing: the
    passages describe the plant, they do not discuss themselves."""
    p = _prompt(_ctx())
    assert "Never mention the texts" in p
    assert "they do not discuss themselves" in p


def test_prompt_forbids_opening_with_soil_or_crop() -> None:
    """All 17 generated queries opened with soil, burying the only words that
    could match a passage."""
    p = _prompt(_ctx())
    assert "Do NOT open with the soil" in p
    assert "first words must be the observed condition" in p


def test_prompt_keeps_soil_available_but_conditional() -> None:
    """Soil is NOT removed — the corpus does link disorder to soil, and the soil
    model is half the multimodal input. It is demoted, not dropped."""
    p = _prompt(_ctx())
    assert "SOIL IS CONDITIONAL, AND NEVER FIRST" in p
    assert "aridity of the soil" in p          # why soil can legitimately belong
    assert "AFTER the symptom" in p
    # the soil reading itself must still reach the model
    assert "Alluvial_Soil" in p and "moderate" in p and "mixed" in p


def test_prompt_lists_corpus_vocabulary_to_prefer_and_avoid() -> None:
    p = _prompt(_ctx())
    assert "WORD CHOICE" in p
    for word in ("yellowness", "paleness", "withering", "oozing"):
        assert word in p, f"missing preferred corpus word: {word}"
    for word in ("necrotic", "lesion", "pathogen", "chlorosis"):
        assert word in p, f"missing banned modern word: {word}"


def test_prompt_shows_worked_examples_including_the_observed_failure() -> None:
    """Few-shot beats instruction alone: the prompt already SAID lead with the
    symptom and Llama ignored it, so show it the real failing query."""
    p = _prompt(_ctx())
    assert "EXAMPLES." in p
    assert p.count("GOOD:") >= 3
    assert "In loam soil with moderate moisture" in p     # the actual §6o failure
    assert "why it is bad" in p


def test_prompt_still_carries_the_original_symptom_first_instruction() -> None:
    """The new rules extend the old ones; they must not have replaced them."""
    p = _prompt(_ctx())
    assert "LEAD WITH THE SYMPTOM" in p
    assert "Potato leaf late blight" in p


def test_symptom_family_phrases_obey_the_word_list() -> None:
    """Rule 3 hands Llama a ready-made phrase per disease family. Those phrases must
    not use the vocabulary the WORD CHOICE section bans, or the prompt contradicts
    itself — which is exactly what happened: rule 3 said blight -> "spreading brown
    NECROTIC patches" while WORD CHOICE banned "necrotic", and 5 of 17 generated
    queries duly kept the banned word."""
    from src.integration.strategy_llm_mediated import _SYSTEM_INSTRUCTIONS as s
    rule3 = s[s.index("3. The query MUST"):s.index("4. Phrase the symptoms")].lower()
    for word in ("necrotic", "lesion", "pathogen", "chlorosis", "fungal", "bacterial"):
        assert word not in rule3, (
            f"rule 3 supplies '{word}', which WORD CHOICE forbids; the prompt would be "
            "telling Llama to use a word it also tells it to avoid"
        )
