"""Phase 7 pipeline smoke — no GPU, no network, no model load.

Wires a stubbed retriever and a stubbed generator into ``RAGPipeline``
and asserts the answer's shape, the citation-extraction pass, and the
metadata threading from retrieved chunks to ``used_chunk_ids``.
"""

from __future__ import annotations

from src.rag.generator import GenerationResult, extract_citations
from src.rag.pipeline import RAGAnswer, RAGPipeline
from src.rag.retriever import RetrievedChunk


# --------------------------------------------------------------------- #
# Stub retriever + generator
# --------------------------------------------------------------------- #


_FAKE_CHUNKS = [
    RetrievedChunk(
        chunk_id="vrik-a",
        text="Apply paste of butter and clarified ghee to the diseased branch.",
        metadata={
            "source_text": "Vrikshayurveda",
            "chapter": "full",
            "verse_or_section": "160.3",
            "translator": "Nalini Sadhale",
            "book_id": "vrikshayurveda",
            "topic_tags": "plant_care, pest_control",
        },
        score=0.91,
        retriever="reranker",
    ),
    RetrievedChunk(
        chunk_id="brih-b",
        text="If birds twitter at sunrise the clouds will bring rain.",
        metadata={
            "source_text": "Brihat Samhita",
            "chapter": "28",
            "verse_or_section": "section_5",
            "translator": "M. Ramakrishna Bhat",
            "book_id": "brihat_samhita",
            "topic_tags": "rainfall, weather_prediction",
        },
        score=0.83,
        retriever="reranker",
    ),
]


class _StubRetriever:
    """Returns the same fake chunks regardless of the query."""

    def __init__(self, chunks=None) -> None:
        self.chunks = chunks if chunks is not None else _FAKE_CHUNKS
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 5):
        self.calls.append((query, k))
        return list(self.chunks[:k])


class _StubGenerator:
    """Returns a canned ``GenerationResult`` that cites BOTH fake chunks."""

    def __init__(self, answer_template: str | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        # Default answer cites both chunks in the exact format
        # GroundedGenerator's regex expects.
        self.answer_template = answer_template or (
            "1. Apply paste of butter and clarified ghee to the diseased "
            "branch [Vrikshayurveda, ch.full, v.160.3].\n"
            "2. If birds twitter at sunrise the clouds will bring rain "
            "[Brihat Samhita, ch.28, v.section_5]."
        )

    def generate(self, query: str, retrieved_chunks) -> GenerationResult:
        self.calls.append((query, len(retrieved_chunks)))
        from src.rag.generator import _match_citations_to_chunks  # noqa: PLC0415

        citations = extract_citations(self.answer_template)
        used = _match_citations_to_chunks(citations, retrieved_chunks)
        return GenerationResult(
            answer=self.answer_template,
            citations=citations,
            used_chunk_ids=used,
            raw_completion=self.answer_template,
        )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_pipeline_answer_returns_expected_shape() -> None:
    retriever = _StubRetriever()
    generator = _StubGenerator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator, default_k=2)

    result = pipeline.answer("how do I treat a diseased tree?")

    assert isinstance(result, RAGAnswer)
    assert result.answer.startswith("1. Apply paste of butter")
    assert result.citations == [
        "Vrikshayurveda ch.full v.160.3",
        "Brihat Samhita ch.28 v.section_5",
    ]
    assert result.used_chunk_ids == ["vrik-a", "brih-b"]
    assert [c.chunk_id for c in result.retrieved] == ["vrik-a", "brih-b"]


def test_pipeline_passes_query_through_to_retriever_and_generator() -> None:
    retriever = _StubRetriever()
    generator = _StubGenerator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    pipeline.answer("rainfall signs?", k=2)
    assert retriever.calls == [("rainfall signs?", 2)]
    assert generator.calls == [("rainfall signs?", 2)]


def test_used_chunk_ids_omits_chunks_not_cited() -> None:
    """Drop one citation from the answer; the matching chunk_id must drop too."""
    retriever = _StubRetriever()
    # Answer cites ONLY the Vrik chunk.
    generator = _StubGenerator(
        answer_template="Apply ghee to the branch [Vrikshayurveda, ch.full, v.160.3]."
    )
    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    result = pipeline.answer("how do I treat a diseased tree?")
    assert result.used_chunk_ids == ["vrik-a"]
    # The Brihat chunk is still in `retrieved` (the retriever returned it)
    # but is NOT in used_chunk_ids because the answer didn't cite it.
    retrieved_ids = [c.chunk_id for c in result.retrieved]
    assert "brih-b" in retrieved_ids
    assert "brih-b" not in result.used_chunk_ids


def test_to_dict_is_json_friendly() -> None:
    retriever = _StubRetriever()
    generator = _StubGenerator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    out = pipeline.answer("anything").to_dict()
    assert set(out) == {"answer", "citations", "used_chunk_ids", "retrieved"}
    assert isinstance(out["retrieved"], list)
    # Each retrieved entry is a dict (asdict'd RetrievedChunk).
    assert isinstance(out["retrieved"][0], dict)
    assert out["retrieved"][0]["chunk_id"] == "vrik-a"


def test_pipeline_requires_collection_or_retriever() -> None:
    import pytest
    with pytest.raises(ValueError, match="collection or a pre-built retriever"):
        RAGPipeline(collection=None, retriever=None, generator=_StubGenerator())
