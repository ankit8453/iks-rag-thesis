"""Retrieved-chunk highlighting.

Given a generated answer and the chunks it cited, produces a list of
:class:`HighlightedChunk` records that the Streamlit demo can render with
span-level highlights. Powered by sentence-level cosine similarity
between answer sentences and chunk sentences — implementation in Phase 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.retriever import RetrievedChunk


@dataclass
class HighlightedSpan:
    """A character span within a chunk that supports a claim in the answer."""

    start_char: int
    end_char: int
    answer_sentence_index: int
    similarity: float


@dataclass
class HighlightedChunk:
    """A retrieved chunk plus the spans that support the generated answer."""

    chunk: RetrievedChunk
    spans: list[HighlightedSpan] = field(default_factory=list)


def highlight_chunks(answer: str, chunks: list[RetrievedChunk]) -> list[HighlightedChunk]:
    """Compute highlight spans linking answer sentences to chunk spans.

    Parameters
    ----------
    answer : str
        The generated answer text.
    chunks : list[RetrievedChunk]
        The chunks that were passed into the generator.

    Returns
    -------
    list[HighlightedChunk]

    Raises
    ------
    NotImplementedError
        Phase 8 — Week 26.
    """
    raise NotImplementedError("Phase 8 — Week 26: implement chunk-span attribution.")
