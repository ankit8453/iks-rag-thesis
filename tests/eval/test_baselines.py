"""Tests for the Phase 11 baselines.

All run locally with fakes — no corpus, no GPU, no network — so the comparison
scaffolding is verified before the expensive Colab pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.eval import baselines


@dataclass
class _Chunk:
    chunk_id: str
    metadata: dict


# ------------------------------------------------------------------ #
# variants
# ------------------------------------------------------------------ #


def test_keyword_baseline_has_no_semantic_stage() -> None:
    """The 'would plain keyword search do?' control must be BM25 only."""
    kw = baselines.get_variant("keyword_only")
    assert kw.use_sparse is True
    assert kw.use_dense is False
    assert kw.use_reranker is False
    assert kw.is_baseline is True


def test_full_variant_is_the_complete_system() -> None:
    full = baselines.get_variant("full")
    assert (full.use_dense, full.use_sparse, full.use_reranker) == (True, True, True)
    assert full.is_baseline is False


def test_every_variant_keeps_at_least_one_retrieval_stage() -> None:
    """HybridRetriever rejects dense+sparse both off — no variant may do that."""
    for v in baselines.RETRIEVAL_VARIANTS:
        assert v.use_dense or v.use_sparse, v.name


def test_variant_names_are_unique_and_lookup_fails_loudly() -> None:
    names = [v.name for v in baselines.RETRIEVAL_VARIANTS]
    assert len(set(names)) == len(names)
    with pytest.raises(KeyError):
        baselines.get_variant("does_not_exist")


def test_build_retriever_passes_flags_and_drops_unused_models(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeRetriever:
        def __init__(self, collection, **kw):
            captured.update(kw)

    import src.rag.retriever as rmod
    monkeypatch.setattr(rmod, "HybridRetriever", _FakeRetriever)

    baselines.build_retriever(
        object(), baselines.get_variant("keyword_only"),
        embedder="EMB", reranker="RER",
    )
    assert captured["use_sparse"] is True
    assert captured["use_dense"] is False
    # a disabled stage must not be handed a model it would never use
    assert "embedder" not in captured
    assert "reranker" not in captured


def test_build_retriever_keeps_models_for_enabled_stages(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeRetriever:
        def __init__(self, collection, **kw):
            captured.update(kw)

    import src.rag.retriever as rmod
    monkeypatch.setattr(rmod, "HybridRetriever", _FakeRetriever)

    baselines.build_retriever(object(), baselines.get_variant("full"),
                              embedder="EMB", reranker="RER")
    assert captured["embedder"] == "EMB"
    assert captured["reranker"] == "RER"


# ------------------------------------------------------------------ #
# scoring adapter
# ------------------------------------------------------------------ #


def test_chunks_to_pairs_reads_book_id() -> None:
    chunks = [_Chunk("c1", {"book_id": "vrikshayurveda"}),
              _Chunk("c2", {"book_id": "upavanavinoda"})]
    assert baselines.chunks_to_pairs(chunks) == [
        ("c1", "vrikshayurveda"), ("c2", "upavanavinoda")]


def test_chunks_to_pairs_falls_back_to_source_text() -> None:
    """Older metadata has no book_id — must still normalise to a book key."""
    chunks = [_Chunk("c1", {"source_text": "Brihat Samhita"})]
    assert baselines.chunks_to_pairs(chunks) == [("c1", "brihat_samhita")]


# ------------------------------------------------------------------ #
# ungrounded control + refusal
# ------------------------------------------------------------------ #


def test_answer_without_grounding_passes_no_context() -> None:
    seen: dict[str, str] = {}

    class _LLM:
        def complete(self, prompt, max_new_tokens=None):
            seen["prompt"] = prompt
            return "  some ungrounded answer  "

    out = baselines.answer_without_grounding(_LLM(), "white powdery coating on leaves")
    assert out == "some ungrounded answer"
    assert "white powdery coating on leaves" in seen["prompt"]
    # the control must NOT smuggle in retrieved passages
    assert "RETRIEVED" not in seen["prompt"].upper()


def test_refusal_detection_and_rate() -> None:
    refusal = ("The retrieved classical-text passages do not contain enough "
               "information to answer this question.")
    assert baselines.is_refusal(refusal) is True
    assert baselines.is_refusal("Apply a paste of neem [Vrikshayurveda, ch.1]") is False
    assert baselines.is_refusal("") is False
    assert baselines.refusal_rate([refusal, "real answer"]) == pytest.approx(0.5)
    assert baselines.refusal_rate([]) == 0.0
