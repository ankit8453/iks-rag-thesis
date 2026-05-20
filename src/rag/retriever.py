"""Dense and BM25 retrievers.

Dense retrieval goes through ChromaDB persisted under
``corpus/vector_db/``. BM25 uses :mod:`rank_bm25` over an in-memory list of
tokenized chunks; index is rebuilt on startup since the corpus is small.

The fused hybrid retriever lives in :mod:`src.rag.hybrid`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunker import Chunk
from src.rag.config import RAGConfig
from src.rag.embedder import Embedder


@dataclass
class RetrievedChunk:
    """A chunk plus its retrieval score and provenance flag.

    Attributes
    ----------
    chunk : Chunk
        The chunk itself.
    score : float
        Retriever-specific score (cosine, BM25, or fused).
    retriever : str
        Which retriever surfaced this chunk: ``"dense"``, ``"bm25"``, or
        ``"hybrid"``.
    """

    chunk: Chunk
    score: float
    retriever: str


class DenseRetriever:
    """ChromaDB-backed dense retriever.

    Parameters
    ----------
    config : RAGConfig
    embedder : Embedder
    persist_dir : Path
        ChromaDB persistence directory (typically
        :data:`src.utils.paths.VECTOR_DB_DIR`).
    """

    def __init__(self, config: RAGConfig, embedder: Embedder, persist_dir: "object") -> None:
        self.config = config
        self.embedder = embedder
        self.persist_dir = persist_dir

    def index(self, chunks: list[Chunk]) -> None:
        """Embed and persist ``chunks`` into the vector store.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement DenseRetriever.index.")

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity.

        Parameters
        ----------
        query : str
            Free-text query.
        top_k : int, optional
            Defaults to ``config.top_k_dense``.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement DenseRetriever.search.")


class BM25Retriever:
    """Sparse BM25 retriever via :mod:`rank_bm25`.

    Parameters
    ----------
    config : RAGConfig
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._index: object | None = None  # Phase 4: rank_bm25.BM25Okapi

    def index(self, chunks: list[Chunk]) -> None:
        """Tokenise and build the BM25 index.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement BM25Retriever.index.")

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks by BM25 score.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement BM25Retriever.search.")
