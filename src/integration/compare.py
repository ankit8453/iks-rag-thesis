"""Side-by-side runner + qualitative reader for Phase 8's three strategies.

Phase 11 will run a rigorous RAGAS context_precision / context_recall
ablation over a curated gold-query set. Phase 8 ships only the
machinery + a qualitative read so the supervisor demo can show A vs B
vs C with the same multimodal input across multiple samples. This
module is deliberately thin: any heavy lifting (HF auth, model load,
ChromaDB rebuild) is the notebook's job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.integration.context import MultimodalContext
from src.integration.strategy_llm_mediated import LLMMediatedStrategy
from src.integration.strategy_multimodal_embedding import (
    MultimodalEmbeddingStrategy,
    MultimodalProjector,
)
from src.integration.strategy_template import TemplateStrategy
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class StrategyResult:
    """One strategy's output for one sample.

    Attributes
    ----------
    strategy : str
        ``"template"`` / ``"llm_mediated"`` / ``"embedding_projection"``.
    query : str
        The retrieval query string (Strategies A and B). For Strategy C
        this is a placeholder string; the actual query is a vector.
    retrieved_chunk_ids : list[str]
        Chunk IDs returned by the retriever, in retrieval order.
    retrieved_sources : list[str]
        Per-hit ``"<source_text> ch.<chapter> v.<verse>"`` summary.
    answer : str | None
        Final grounded answer if the strategy produced one (Strategies
        A and B). ``None`` for Strategy C (retrieval-only).
    citations : list[str]
        Citations parsed out of ``answer`` if present.
    """

    strategy: str
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_sources: list[str]
    answer: str | None = None
    citations: list[str] | None = None


def _format_sources(retrieved: list[Any]) -> list[str]:
    """Render retrieved-chunk metadata into a compact display string."""
    out: list[str] = []
    for h in retrieved:
        # h is either RetrievedChunk-like (.metadata) or a dict (Strategy C).
        meta = getattr(h, "metadata", None)
        if meta is None and isinstance(h, dict):
            meta = h.get("metadata") or {}
        meta = meta or {}
        src = (
            f"{meta.get('source_text', '?')} ch.{meta.get('chapter', '?')} "
            f"v.{meta.get('verse_or_section', '?')}"
        )
        out.append(src)
    return out


def _chunk_ids_of(retrieved: list[Any]) -> list[str]:
    out: list[str] = []
    for h in retrieved:
        cid = getattr(h, "chunk_id", None)
        if cid is None and isinstance(h, dict):
            cid = h.get("chunk_id")
        out.append(str(cid) if cid is not None else "")
    return out


def _run_template(
    ctx: MultimodalContext,
    rag_pipeline: Any,
    template: TemplateStrategy,
    k: int,
    answer: bool,
) -> StrategyResult:
    query = template.build_query(ctx)
    if answer:
        rag_answer = rag_pipeline.answer(query, k=k)
        retrieved = list(rag_answer.retrieved)
        return StrategyResult(
            strategy="template",
            query=query,
            retrieved_chunk_ids=_chunk_ids_of(retrieved),
            retrieved_sources=_format_sources(retrieved),
            answer=rag_answer.answer,
            citations=list(rag_answer.citations or []),
        )
    retrieved = list(rag_pipeline.retriever.retrieve(query, k=k))
    return StrategyResult(
        strategy="template",
        query=query,
        retrieved_chunk_ids=_chunk_ids_of(retrieved),
        retrieved_sources=_format_sources(retrieved),
    )


def _run_llm(
    ctx: MultimodalContext,
    rag_pipeline: Any,
    llm_strategy: LLMMediatedStrategy,
    k: int,
    answer: bool,
) -> StrategyResult:
    # Phase 8 reuses the SAME Llama instance the Phase 7 RAG pipeline
    # already loaded — no second model load.
    llm = getattr(rag_pipeline, "generator", None)
    if llm is None:
        raise RuntimeError(
            "rag_pipeline.generator is None — Strategy B needs the "
            "Phase 7 GroundedGenerator on the pipeline."
        )
    query = llm_strategy.build_query(ctx, llm)
    if answer:
        rag_answer = rag_pipeline.answer(query, k=k)
        retrieved = list(rag_answer.retrieved)
        return StrategyResult(
            strategy="llm_mediated",
            query=query,
            retrieved_chunk_ids=_chunk_ids_of(retrieved),
            retrieved_sources=_format_sources(retrieved),
            answer=rag_answer.answer,
            citations=list(rag_answer.citations or []),
        )
    retrieved = list(rag_pipeline.retriever.retrieve(query, k=k))
    return StrategyResult(
        strategy="llm_mediated",
        query=query,
        retrieved_chunk_ids=_chunk_ids_of(retrieved),
        retrieved_sources=_format_sources(retrieved),
    )


def _run_embedding(
    ctx: MultimodalContext,
    embed_strategy: MultimodalEmbeddingStrategy,
    projector: MultimodalProjector,
    chroma_collection: Any,
    embedder: Any,
    k: int,
) -> StrategyResult:
    retrieved = embed_strategy.retrieve_via_embedding(
        ctx, projector, chroma_collection, embedder, k=k,
    )
    # Strategy C deliberately does NOT generate an answer — the
    # comparison is about retrieval modality, not generation. Generating
    # would require constructing a fake query string, which would muddy
    # the ablation read.
    return StrategyResult(
        strategy="embedding_projection",
        query="[multimodal vector — no text query]",
        retrieved_chunk_ids=_chunk_ids_of(retrieved),
        retrieved_sources=_format_sources(retrieved),
        answer=None,
        citations=None,
    )


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #


def run_all_strategies(
    ctx: MultimodalContext,
    rag_pipeline: Any,
    *,
    projector: MultimodalProjector | None = None,
    chroma_collection: Any | None = None,
    embedder: Any | None = None,
    k: int = 5,
    answer: bool = True,
    template_strategy: TemplateStrategy | None = None,
    llm_strategy: LLMMediatedStrategy | None = None,
    embed_strategy: MultimodalEmbeddingStrategy | None = None,
) -> dict[str, StrategyResult]:
    """Run Strategies A / B / C on one ``ctx`` and return per-strategy results.

    Parameters
    ----------
    ctx
        The :class:`MultimodalContext` produced by
        :func:`~src.integration.context.build_multimodal_context`.
    rag_pipeline
        The Phase 7 :class:`~src.rag.pipeline.RAGPipeline`. Used by A
        (to retrieve + generate) and B (same, plus its ``generator``
        attribute is reused as Strategy B's LLM).
    projector, chroma_collection, embedder
        Required if Strategy C should be run. If any is ``None``,
        Strategy C is silently skipped (and not included in the
        returned dict).
    k
        Top-k retrieval per strategy.
    answer
        If True, also call the RAG generator for Strategies A + B.
        If False, retrieve only.
    template_strategy, llm_strategy, embed_strategy
        Allow injecting custom configs. If ``None``, default-config
        instances are constructed.

    Returns
    -------
    dict[str, StrategyResult]
        Keyed by ``"template"`` / ``"llm_mediated"`` / ``"embedding_projection"``.
    """
    from src.integration.config import (
        LLMMediatedStrategyConfig,
        MultimodalEmbeddingStrategyConfig,
        TemplateStrategyConfig,
    )

    template = template_strategy or TemplateStrategy(TemplateStrategyConfig())
    llmstrat = llm_strategy or LLMMediatedStrategy(LLMMediatedStrategyConfig())

    results: dict[str, StrategyResult] = {}

    # ---- Strategy A --------------------------------------------- #
    _LOGGER.info("Strategy A (template) running ...")
    results["template"] = _run_template(ctx, rag_pipeline, template, k=k, answer=answer)

    # ---- Strategy B --------------------------------------------- #
    _LOGGER.info("Strategy B (LLM-mediated) running ...")
    results["llm_mediated"] = _run_llm(ctx, rag_pipeline, llmstrat, k=k, answer=answer)

    # ---- Strategy C --------------------------------------------- #
    if projector is not None and chroma_collection is not None and embedder is not None:
        embstrat = embed_strategy or MultimodalEmbeddingStrategy(
            MultimodalEmbeddingStrategyConfig()
        )
        _LOGGER.info("Strategy C (embedding-projection ablation) running ...")
        results["embedding_projection"] = _run_embedding(
            ctx, embstrat, projector, chroma_collection, embedder, k=k,
        )
    else:
        _LOGGER.info(
            "Strategy C skipped: projector / chroma_collection / embedder "
            "not provided."
        )

    return results


def qualitative_compare(
    results_per_sample: list[dict[str, StrategyResult]],
    relevant_book_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Render a side-by-side qualitative comparison row per (sample, strategy).

    Phase 8 is QUALITATIVE — no precision / recall scores. The "on-topic
    overlap count" here is just *"how many of the k retrieved chunks
    came from a plausibly relevant book"*, where the plausibly-relevant
    book list is supplied by the caller (or defaults to all 4 IKS
    books). It is a heuristic for the supervisor demo; the rigorous
    RAGAS read happens in Phase 11.

    Parameters
    ----------
    results_per_sample
        One ``dict`` per sample, as returned by :func:`run_all_strategies`.
    relevant_book_ids
        The list of ``source_text`` strings that count as on-topic for
        this comparison. Defaults to the full 4-book corpus.

    Returns
    -------
    list[dict]
        Flat list of rows, each carrying
        ``{sample_idx, strategy, query, retrieved_sources, on_topic_count}``.
    """
    if relevant_book_ids is None:
        relevant_book_ids = [
            "Vrikshayurveda",
            "Brihat Samhita",
            "Krishi Parashara",
            "Upavanavinoda",
        ]
    relevant_set = set(s.lower() for s in relevant_book_ids)

    rows: list[dict[str, Any]] = []
    for idx, per_strategy in enumerate(results_per_sample):
        for strat_name, r in per_strategy.items():
            on_topic = sum(
                1 for s in r.retrieved_sources
                if any(book in s.lower() for book in relevant_set)
            )
            rows.append({
                "sample_idx": idx,
                "strategy": strat_name,
                "query": r.query,
                "retrieved_sources": list(r.retrieved_sources),
                "on_topic_count": on_topic,
                "k": len(r.retrieved_sources),
            })
    return rows


__all__ = [
    "StrategyResult",
    "qualitative_compare",
    "run_all_strategies",
]
