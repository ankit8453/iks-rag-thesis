"""Hybrid dense + BM25 retriever.

Fuses scores from :class:`~src.rag.retriever.DenseRetriever` and
:class:`~src.rag.retriever.BM25Retriever` via ``hybrid_alpha``-weighted
sum of min-max normalised scores. The fused candidates are passed to
:class:`~src.rag.reranker.CrossEncoderReranker` for the final top-k.
"""

from __future__ import annotations

from src.rag.config import RAGConfig
from src.rag.retriever import BM25Retriever, DenseRetriever, RetrievedChunk


class HybridRetriever:
    """Dense + BM25 fused retriever.

    Parameters
    ----------
    config : RAGConfig
    dense : DenseRetriever
    bm25 : BM25Retriever
    """

    def __init__(
        self,
        config: RAGConfig,
        dense: DenseRetriever,
        bm25: BM25Retriever,
    ) -> None:
        self.config = config
        self.dense = dense
        self.bm25 = bm25

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k fused chunks.

        Parameters
        ----------
        query : str
        top_k : int, optional
            Defaults to ``max(config.top_k_dense, config.top_k_bm25)``.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement HybridRetriever.search.")
