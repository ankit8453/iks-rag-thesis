"""Cross-encoder reranker (BAAI/bge-reranker-base).

Takes the fused candidate set from :class:`HybridRetriever` and re-scores
each ``(query, chunk)`` pair with a cross-encoder. Returns the top-k by
the reranker's score — these are the chunks fed into the generator's
prompt template.
"""

from __future__ import annotations

from src.rag.config import RAGConfig
from src.rag.retriever import RetrievedChunk


class CrossEncoderReranker:
    """Wrapper around :mod:`sentence_transformers.CrossEncoder`.

    Parameters
    ----------
    config : RAGConfig
        Provides ``reranker_model``.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._model: object | None = None  # Phase 4: CrossEncoder

    def load(self) -> None:
        """Load the cross-encoder weights.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: load BGE reranker.")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks by cross-encoder score.

        Parameters
        ----------
        query : str
        candidates : list[RetrievedChunk]
        top_k : int, optional
            Defaults to ``config.top_k_rerank``.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement CrossEncoderReranker.rerank.")
