"""Regression tests for the two Phase 8 bugs caught during the first
Colab run:

- **Bug 1** — :class:`DiseaseInferenceEngine` was defaulting class names
  to ``"class_<i>"`` when ``class_names`` was not passed, so Strategy A
  queries read *"Organic treatment for class 0 affecting rice ..."*.
  The Phase-5 PlantDoc training stores its class map at
  ``data/splits/plantdoc/class_map.json``; the engine should now
  auto-resolve it whenever the model source's identifier contains
  ``plantdoc``, and :func:`build_multimodal_context` should refuse to
  return a context whose ``disease_pred.class_name`` still matches the
  ``^class_\\d+$`` placeholder pattern.
- **Bug 2** — :class:`LLMMediatedStrategy._invoke_llm` was trying
  ``.generate(prompt)`` first, which raised ``TypeError`` on Phase 7's
  :class:`GroundedGenerator` (its ``generate(query, retrieved_chunks)``
  signature requires two args). Strategy B must now try ``.complete``
  before ``.generate`` so the plain prompt → text path is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.disease.infer import _load_class_names_for_source
from src.integration.causation import CausalContext, CausalPathway
from src.integration.config import LLMMediatedStrategyConfig
from src.integration.context import _INDEX_LABEL_RE, MultimodalContext
from src.integration.strategy_llm_mediated import LLMMediatedStrategy


# --------------------------------------------------------------------- #
# Bug 1
# --------------------------------------------------------------------- #


def test_disease_class_names_auto_resolve_for_plantdoc() -> None:
    """Bug 1 — PlantDoc has 27 classes; the loader should return them
    in index order, with "Apple Scab Leaf" at index 0."""
    names = _load_class_names_for_source(
        model_source="ankit-iiitdmj/iks-disease-plantdoc",
        num_classes=27,
    )
    assert names is not None
    assert len(names) == 27
    assert names[0] == "Apple Scab Leaf"
    # Sanity: every entry is a non-empty string and none are placeholders.
    for n in names:
        assert isinstance(n, str) and n.strip()
        assert not _INDEX_LABEL_RE.match(n), (
            f"class_map.json entry looks like a placeholder: {n!r}"
        )


def test_disease_class_names_return_none_on_mismatch() -> None:
    """If the resolved class_map.json doesn't agree with the checkpoint's
    num_classes, the loader must refuse (None) rather than silently
    mis-label."""
    names = _load_class_names_for_source(
        model_source="ankit-iiitdmj/iks-disease-plantdoc",
        num_classes=999,
    )
    assert names is None


def test_disease_class_names_unknown_source_returns_none() -> None:
    """No PlantDoc / PlantVillage / PaddyDoctor substring → no mapping."""
    names = _load_class_names_for_source(
        model_source="some-other-org/unrelated-model",
        num_classes=27,
    )
    assert names is None


def test_index_label_regex_matches_placeholders_only() -> None:
    """Sanity for the guard pattern used inside build_multimodal_context."""
    for placeholder in ("class_0", "class_27", "CLASS_3"):
        assert _INDEX_LABEL_RE.match(placeholder), placeholder
    for real in ("Tomato leaf yellow virus", "Apple Scab Leaf", "Rice___Blast"):
        assert not _INDEX_LABEL_RE.match(real), real


def test_build_multimodal_context_rejects_placeholder_disease_label() -> None:
    """End-to-end Bug-1 guard: even if both engines are passed in
    pre-built (so the loader logic never runs), a context whose
    disease prediction carries a ``class_<i>`` label must not be
    silently returned — :func:`build_multimodal_context` must raise."""
    from src.integration.context import build_multimodal_context

    @dataclass
    class _StubDiseasePred:
        class_name: str = "class_3"
        confidence: float = 0.9
        class_index: int = 3
        logits: list[float] | None = None

    class _StubInferenceResult:
        def __init__(self, prediction):
            self.prediction = prediction
            self.top_k = None
            self.gradcam_overlay = None

    class _StubDiseaseEngine:
        def predict(self, *_a, **_kw):
            return _StubInferenceResult(prediction=_StubDiseasePred())

        def predict_with_embedding(self, *_a, **_kw):
            import numpy as np
            return (
                _StubInferenceResult(prediction=_StubDiseasePred()),
                np.zeros(1792, dtype="float32"),
            )

    @dataclass
    class _StubSoilPred:
        soil_type: str = "Alluvial_Soil"
        moisture_appearance: str = "moist"
        texture: str = "mixed"
        per_head_confidence: dict | None = None

    class _StubSoilResult:
        def __init__(self, prediction, embedding):
            self.prediction = prediction
            self.embedding = embedding

    class _StubSoilEngine:
        def predict(self, *_a, **kw):
            import numpy as np
            emb = np.zeros(1280, dtype="float32") if kw.get("with_embedding") else None
            return _StubSoilResult(prediction=_StubSoilPred(), embedding=emb)

    # Fake one-pixel image — never decoded by the stub engines.
    from PIL import Image
    img = Image.new("RGB", (8, 8))

    with pytest.raises(ValueError, match="placeholder label"):
        build_multimodal_context(
            leaf_image=img,
            soil_image=img,
            crop_type="rice",
            causal_pathway=CausalPathway.UNKNOWN,
            disease_engine=_StubDiseaseEngine(),
            soil_engine=_StubSoilEngine(),
            capture_embeddings=True,
        )


# --------------------------------------------------------------------- #
# Bug 2
# --------------------------------------------------------------------- #


class _GroundedGeneratorLike:
    """Mimics :class:`GroundedGenerator`'s relevant Phase 8 API surface.

    Has BOTH a Phase-7-style ``generate(query, retrieved_chunks)`` (the
    one that raised ``TypeError`` when called with a single positional
    arg) AND a new ``complete(prompt)`` for the plain-prompt rewrite.
    The bug-fix contract: Strategy B must NOT call ``generate`` at all
    on this object, because ``generate(prompt)`` would now actually run
    on Llama, retrieve nothing, and return a §17-grounded refusal
    instead of a rewritten query.
    """

    def __init__(self) -> None:
        self.complete_calls: list[str] = []
        self.generate_calls: list[tuple] = []

    def complete(self, prompt: str) -> str:
        self.complete_calls.append(prompt)
        return "Find passages on classical remedies for scorched rice leaves."

    def generate(self, query, retrieved_chunks):
        # Two-arg signature — Strategy B must NEVER reach this in the
        # post-fix code (the .complete path takes precedence).
        self.generate_calls.append((query, retrieved_chunks))
        raise AssertionError(
            "Strategy B reached the two-arg .generate(); the .complete "
            "path is supposed to win."
        )


def _stub_context() -> MultimodalContext:
    @dataclass
    class _D:
        class_name: str = "Tomato leaf yellow virus"
        confidence: float = 0.81
        class_index: int = 0
        logits: list[float] | None = None

    @dataclass
    class _S:
        soil_type: str = "Alluvial_Soil"
        moisture_appearance: str = "moist"
        texture: str = "fine"
        per_head_confidence: dict | None = None

    return MultimodalContext(
        disease_pred=_D(),
        soil_pred=_S(),
        crop_type="rice",
        causal_context=CausalContext(pathway=CausalPathway.SOIL_DRIVEN),
    )


def test_strategy_b_prefers_complete_over_generate_for_grounded_generator() -> None:
    llm = _GroundedGeneratorLike()
    strat = LLMMediatedStrategy(LLMMediatedStrategyConfig())
    query = strat.build_query(_stub_context(), llm)

    # Strategy B used .complete (returned the stub's rewritten sentence).
    assert llm.complete_calls and not llm.generate_calls, (
        "Bug 2 regression: Strategy B should call .complete first, NOT "
        ".generate(query, retrieved_chunks)."
    )
    assert "scorched rice leaves" in query.lower()


def test_strategy_b_falls_back_to_generate_when_no_complete() -> None:
    """If a future generator only exposes .generate(prompt) (the
    common HF / OpenAI shape), Strategy B must still work."""

    class _PlainHFGen:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return "Find classical passages on whitish leaf lesions in tomato."

    llm = _PlainHFGen()
    strat = LLMMediatedStrategy(LLMMediatedStrategyConfig())
    query = strat.build_query(_stub_context(), llm)
    assert llm.calls
    assert "whitish leaf lesions" in query.lower()


def test_strategy_b_clear_error_when_llm_has_no_usable_method() -> None:
    """A bare object with no LLM-like methods should fail loudly."""

    class _Nothing:
        pass

    strat = LLMMediatedStrategy(LLMMediatedStrategyConfig())
    with pytest.raises(TypeError, match="no usable"):
        strat.build_query(_stub_context(), _Nothing())
