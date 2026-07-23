"""Phase 11 baselines — what the system is measured *against*.

A retrieval score on its own says little; the contribution is only demonstrated
by comparison. Two comparisons matter, and they answer different objections:

* **"Does the semantic bridge actually help, or would plain keyword search do?"**
  → ``keyword_only``: BM25 alone, no dense retrieval, no reranking.
* **"Do you need the classical corpus at all, or does a large model already
  know this?"** → :func:`answer_without_grounding`: the same LLM asked directly,
  with no retrieved passages. It cannot cite, which is exactly the point.

Two further variants (``dense_only``, ``hybrid_no_rerank``) are the §27 retrieval
ablations — they isolate which stage earns its keep.

Every retrieval variant is a *configuration* of the existing
:class:`~src.rag.retriever.HybridRetriever`, not a reimplementation, so a
baseline can never drift from the system it is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The locked refusal sentence from the §17 grounded-advisor prompt. Used to
#: detect an honest refusal on the deliberately-unanswerable queries.
REFUSAL_MARKER: str = "do not contain enough"

UNGROUNDED_PROMPT: str = (
    "You are an agricultural advisor. Answer the following question about a "
    "plant problem in a few sentences.\n\nQuestion: {query}\n\nAnswer:"
)


@dataclass(frozen=True)
class RetrievalVariant:
    """One retrieval configuration to evaluate."""

    name: str
    description: str
    use_dense: bool
    use_sparse: bool
    use_reranker: bool
    is_baseline: bool = False


#: The system plus everything it is compared against.
RETRIEVAL_VARIANTS: tuple[RetrievalVariant, ...] = (
    RetrievalVariant(
        "full", "Our system: dense + BM25, fused, then cross-encoder reranked",
        use_dense=True, use_sparse=True, use_reranker=True,
    ),
    RetrievalVariant(
        "keyword_only", "Baseline: BM25 keyword search only — no semantic bridge",
        use_dense=False, use_sparse=True, use_reranker=False, is_baseline=True,
    ),
    RetrievalVariant(
        "dense_only", "Ablation: dense semantic search only, no keyword stage",
        use_dense=True, use_sparse=False, use_reranker=False,
    ),
    RetrievalVariant(
        "hybrid_no_rerank", "Ablation: dense + BM25 fused, but no reranking",
        use_dense=True, use_sparse=True, use_reranker=False,
    ),
)


def get_variant(name: str) -> RetrievalVariant:
    for v in RETRIEVAL_VARIANTS:
        if v.name == name:
            return v
    raise KeyError(f"unknown retrieval variant {name!r}; "
                   f"expected one of {[v.name for v in RETRIEVAL_VARIANTS]}")


def build_retriever(collection: Any, variant: RetrievalVariant, **shared: Any) -> Any:
    """Construct a ``HybridRetriever`` configured for ``variant``.

    ``shared`` forwards pre-loaded models (``embedder``, ``reranker``) and depth
    settings so every variant reuses one embedder/reranker instead of reloading
    them per run.
    """
    from src.rag.retriever import HybridRetriever  # noqa: PLC0415

    # A stage that is off must not receive its model, or the retriever would
    # load a model it never uses.
    if not variant.use_dense:
        shared.pop("embedder", None)
    if not variant.use_reranker:
        shared.pop("reranker", None)

    return HybridRetriever(
        collection,
        use_dense=variant.use_dense,
        use_sparse=variant.use_sparse,
        use_reranker=variant.use_reranker,
        **shared,
    )


def chunks_to_pairs(chunks: list[Any]) -> list[tuple[str, str]]:
    """``RetrievedChunk`` list → ``(chunk_id, book_id)`` pairs for scoring.

    Falls back to ``source_text`` (lower-cased, spaces → underscores) when
    ``book_id`` is absent, so scoring still works on older chunk metadata.
    """
    pairs: list[tuple[str, str]] = []
    for c in chunks:
        meta = getattr(c, "metadata", {}) or {}
        book = meta.get("book_id") or meta.get("source_text") or ""
        pairs.append((getattr(c, "chunk_id", ""), str(book).strip().lower().replace(" ", "_")))
    return pairs


def answer_without_grounding(llm: Any, query: str, *, max_new_tokens: int = 256) -> str:
    """Baseline: ask the LLM directly, with **no** retrieved passages.

    This is the "do we need the corpus?" control. The answer has no sources to
    cite, so citation verification on it should find nothing to verify — which
    is the contrast the grounded system is meant to win on.
    """
    return llm.complete(UNGROUNDED_PROMPT.format(query=query),
                        max_new_tokens=max_new_tokens).strip()


def is_refusal(answer: str) -> bool:
    """True when the answer is the locked "insufficient evidence" refusal.

    On the deliberately-unanswerable queries, refusing is the *correct* outcome,
    so this is scored as a success rather than a failure.
    """
    return REFUSAL_MARKER.lower() in (answer or "").lower()


def refusal_rate(answers: list[str]) -> float:
    """Fraction of answers that were honest refusals."""
    if not answers:
        return 0.0
    return sum(1 for a in answers if is_refusal(a)) / len(answers)


__all__ = [
    "REFUSAL_MARKER",
    "RETRIEVAL_VARIANTS",
    "RetrievalVariant",
    "answer_without_grounding",
    "build_retriever",
    "chunks_to_pairs",
    "get_variant",
    "is_refusal",
    "refusal_rate",
]
