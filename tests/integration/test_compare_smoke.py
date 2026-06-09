"""Phase 8 — ``compare.run_all_strategies`` smoke test.

Stub RAG pipeline + stub vision context (NO real models, no GPU, no
network). Validates:

- ``run_all_strategies`` returns a dict keyed by strategy name.
- Each :class:`StrategyResult` has the expected fields populated.
- The causal-context value threads through into Strategy A's query.
- Strategy C is skipped (silently) when ``projector`` is not supplied,
  matching the contract documented in :func:`run_all_strategies`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.integration import CausalContext, CausalPathway, MultimodalContext
from src.integration.compare import (
    StrategyResult,
    qualitative_compare,
    run_all_strategies,
)


# --------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------- #


@dataclass
class _StubDiseasePred:
    class_name: str = "Rice___Blast_Disease"
    confidence: float = 0.81
    class_index: int = 0
    logits: list[float] | None = None


@dataclass
class _StubSoilPred:
    soil_type: str = "Black_Soil"
    moisture_appearance: str = "moist"
    texture: str = "fine"
    per_head_confidence: dict | None = None


@dataclass
class _StubRetrievedChunk:
    chunk_id: str
    metadata: dict
    text: str = "stub chunk body"
    score: float = 0.85
    retriever: str = "stub"


class _StubRetriever:
    """Returns a fixed list of stub chunks regardless of query."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str, k: int = 5) -> list[_StubRetrievedChunk]:
        self.calls.append(query)
        # Mix the 4 corpus books so qualitative_compare's on-topic
        # heuristic has something to count.
        books = ["Vrikshayurveda", "Brihat Samhita", "Krishi Parashara", "Upavanavinoda"]
        return [
            _StubRetrievedChunk(
                chunk_id=f"chunk_{i}",
                metadata={
                    "source_text": books[i % 4],
                    "chapter": str(i),
                    "verse_or_section": str(i),
                },
            )
            for i in range(k)
        ]


@dataclass
class _StubAnswer:
    answer: str
    retrieved: list[_StubRetrievedChunk]
    citations: list[str]
    used_chunk_ids: list[str]


class _StubGenerator:
    """A bare-bones LLM stub for Strategy B."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # Return a one-sentence "rewritten" query so Strategy B has
        # something concrete to retrieve with.
        return (
            "Find passages on classical remedies for rice plants suffering "
            "scorched leaves grown in dark moist heavy soil."
        )


class _StubRAGPipeline:
    def __init__(self) -> None:
        self.retriever = _StubRetriever()
        self.generator = _StubGenerator()
        self.answer_calls: list[str] = []

    def answer(self, query: str, k: int = 5) -> _StubAnswer:
        self.answer_calls.append(query)
        retrieved = self.retriever.retrieve(query, k=k)
        return _StubAnswer(
            answer=(
                "Apply organic mulch around the affected plants. "
                "[Vrikshayurveda, ch.1, v.4]"
            ),
            retrieved=retrieved,
            citations=["[Vrikshayurveda, ch.1, v.4]"],
            used_chunk_ids=[retrieved[0].chunk_id],
        )


def _make_ctx(pathway: CausalPathway = CausalPathway.SOIL_DRIVEN) -> MultimodalContext:
    return MultimodalContext(
        disease_pred=_StubDiseasePred(),
        soil_pred=_StubSoilPred(),
        crop_type="rice",
        causal_context=CausalContext(pathway=pathway, notes="Field waterlogged."),
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_run_all_strategies_shape_without_projector() -> None:
    pipeline = _StubRAGPipeline()
    results = run_all_strategies(
        _make_ctx(), pipeline, projector=None, k=4, answer=True,
    )
    # Strategy C is silently skipped when projector is missing.
    assert set(results) == {"template", "llm_mediated"}
    for key, r in results.items():
        assert isinstance(r, StrategyResult)
        assert r.strategy == key
        assert isinstance(r.query, str) and r.query, f"{key}: empty query"
        assert len(r.retrieved_chunk_ids) == 4
        assert len(r.retrieved_sources) == 4
        # answer=True so both A and B should have populated answers.
        assert r.answer is not None
        assert r.citations and "Vrikshayurveda" in r.citations[0]


def test_causal_context_threads_into_template_query() -> None:
    """A SOIL_DRIVEN pathway should produce a 'soil restoration' clause
    in Strategy A's query — that's how C5 manifests downstream."""
    pipeline = _StubRAGPipeline()
    results = run_all_strategies(
        _make_ctx(CausalPathway.SOIL_DRIVEN), pipeline, projector=None, k=3, answer=False,
    )
    q = results["template"].query.lower()
    assert "soil restoration" in q, f"expected soil clause; got: {q!r}"


def test_causal_context_unknown_yields_no_clause() -> None:
    pipeline = _StubRAGPipeline()
    results = run_all_strategies(
        _make_ctx(CausalPathway.UNKNOWN), pipeline, projector=None, k=3, answer=False,
    )
    q = results["template"].query.lower()
    for keyword in ("soil restoration", "pest", "spread"):
        assert keyword not in q


def test_strategy_b_uses_pipeline_generator() -> None:
    """Strategy B must reuse ``pipeline.generator`` (Phase 7 Llama)."""
    pipeline = _StubRAGPipeline()
    results = run_all_strategies(_make_ctx(), pipeline, projector=None, k=2, answer=False)
    # The stub generator recorded at least one prompt.
    assert pipeline.generator.prompts, "Strategy B never invoked the LLM"
    # And the LLM output (the stub's fixed sentence) ended up as the query.
    assert "scorched leaves" in results["llm_mediated"].query.lower()


def test_qualitative_compare_counts_on_topic_hits() -> None:
    """The on-topic heuristic counts retrieved chunks whose source_text
    matches one of the relevant books — defaults to all 4 IKS books."""
    pipeline = _StubRAGPipeline()
    results = run_all_strategies(_make_ctx(), pipeline, projector=None, k=4, answer=False)
    rows = qualitative_compare([results])
    assert {r["strategy"] for r in rows} == {"template", "llm_mediated"}
    for row in rows:
        # Every stub chunk is one of the 4 corpus books → all should be on-topic.
        assert row["on_topic_count"] == row["k"] == 4
        assert row["sample_idx"] == 0


def test_qualitative_compare_filters_by_relevant_book_list() -> None:
    pipeline = _StubRAGPipeline()
    results = run_all_strategies(_make_ctx(), pipeline, projector=None, k=4, answer=False)
    rows = qualitative_compare([results], relevant_book_ids=["Krishi Parashara"])
    for row in rows:
        # The stub retriever round-robins books, so only 1 of every 4 chunks
        # comes from Krishi Parashara.
        assert row["on_topic_count"] == 1


def test_run_all_strategies_raises_if_pipeline_has_no_generator() -> None:
    """Strategy B explicitly errors when the pipeline lacks a generator."""

    class _NoGenPipeline:
        retriever = _StubRetriever()
        generator: Any = None

        def answer(self, query: str, k: int = 5) -> Any:
            raise AssertionError("should not be reached")

    with pytest.raises(RuntimeError, match="generator is None"):
        run_all_strategies(_make_ctx(), _NoGenPipeline(), projector=None, k=2, answer=False)
