"""Embedding wrapper.

Wraps :mod:`sentence_transformers` so the rest of the codebase only sees
``Embedder.encode(text) -> ndarray``. Default model is BAAI/bge-large-en;
``RAGConfig.multilingual_fallback`` is used when source language differs.
"""

from __future__ import annotations

from src.rag.config import RAGConfig


class Embedder:
    """Sentence-Transformers wrapper.

    Parameters
    ----------
    config : RAGConfig
        Provides ``embedding_model`` and (optionally)
        ``multilingual_fallback``.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._model: object | None = None  # Phase 4: SentenceTransformer
        self._dim: int | None = None

    def load(self) -> None:
        """Lazy-load the sentence-transformers model.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 13.
        """
        raise NotImplementedError("Phase 4 — Week 13: load sentence-transformers model.")

    def encode(self, texts: list[str]) -> "object":
        """Encode a batch of strings into a ``(n, dim)`` numpy array.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 13.
        """
        raise NotImplementedError("Phase 4 — Week 13: implement Embedder.encode.")

    @property
    def dim(self) -> int:
        """Embedding dimensionality.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        """
        if self._dim is None:
            raise RuntimeError("Embedder.load() must be called before .dim is available.")
        return self._dim
