"""Chunking strategies for the IKS corpus.

Two strategies are planned (selected by :class:`~src.rag.config.ChunkerConfig`):

1. ``paragraph`` — split on paragraph breaks; merge until ``target_tokens``.
   Default for classical-text translations where paragraphs are short.
2. ``sliding_window`` — fixed-length token windows with
   ``overlap_tokens`` of overlap. Default for long, unstructured prose.

Each chunk carries enough metadata that downstream retrieval and citation
verification (contribution C4) can trace a generated claim back to its
source line range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.config import ChunkerConfig


@dataclass
class Chunk:
    """One retrievable text chunk.

    Attributes
    ----------
    chunk_id : str
        Stable, globally-unique ID. Format ``<source_id>:<index>``.
    source_id : str
        Identifier of the parent :class:`~src.rag.corpus_loader.IKSDocument`.
    text : str
        Chunk body.
    start_char : int
        Inclusive character offset within the cleaned source.
    end_char : int
        Exclusive character offset within the cleaned source.
    token_count : int
        Approximate token count (model-agnostic estimator).
    metadata : dict[str, str]
        Inherits from the parent document; useful for filtered retrieval.
    """

    chunk_id: str
    source_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


class Chunker:
    """Strategy-dispatched chunker.

    Parameters
    ----------
    config : ChunkerConfig
        Selects strategy and chunk size.
    """

    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config

    def chunk(self, source_id: str, text: str) -> list[Chunk]:
        """Split ``text`` into a list of :class:`Chunk` objects.

        Parameters
        ----------
        source_id : str
            Identifier of the parent document.
        text : str
            Cleaned, plain-text body.

        Returns
        -------
        list[Chunk]

        Raises
        ------
        NotImplementedError
            Implementation deferred to Phase 4 — Week 13.
        """
        if self.config.strategy == "paragraph":
            return self._paragraph_chunk(source_id, text)
        return self._sliding_window_chunk(source_id, text)

    def _paragraph_chunk(self, source_id: str, text: str) -> list[Chunk]:
        raise NotImplementedError("Phase 4 — Week 13: paragraph chunker.")

    def _sliding_window_chunk(self, source_id: str, text: str) -> list[Chunk]:
        raise NotImplementedError("Phase 4 — Week 13: sliding-window chunker.")
