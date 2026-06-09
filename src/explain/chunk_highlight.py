"""Retrieved-chunk highlighting (Phase 9 §C).

Purely lexical, no model needed: tokenize the query and the chunk text
with the same simple normaliser, intersect the token sets, and wrap
matched terms in the chunk with ``**…**`` markers so the Streamlit /
notebook caller can render them. This is intentionally simple — the
goal is *audit*-grade transparency for the supervisor demo, not a
state-of-the-art rationale extractor.

Why not use the answer text (as the earlier scaffold did)? Phase 9
explains the RETRIEVAL step. Aligning chunks to the LLM's answer
sentences mixes two failure modes (bad retrieval vs bad generation)
into one signal. Aligning chunks to the QUERY isolates retrieval and
makes "why was this chunk chosen?" answerable end-to-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Tight English stopword set — covers the words that recur in
# template queries ("organic", "treatment", "for", "in", ...) without
# pulling in nltk. Kept conservative: removing too much would mask
# whether the chunk truly matches the intent of the query.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "with", "which", "what", "how", "should", "be",
    "appears", "appearing", "affecting", "growing", "grown",
    "treatment", "organic",
    # Soil noise-words that are template artefacts, not signal:
    "soil", "type",
    # Common verb fillers in the template:
    "applied", "apply", "used", "use", "using",
})

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> set[str]:
    """Lower-case, drop punctuation + stopwords, return a token set.

    Two simple normalisations to bridge query/chunk vocabulary mismatch:

    - Lower-case everything (template emits ``Tomato leaf...``, chunk
      may say ``tomato``).
    - Strip a trailing ``s`` from words ≥ 4 chars long. Crude singular
      normalisation that helps ``leaves`` ↔ ``leaf`` and ``trees`` ↔
      ``tree`` without dragging in a stemmer.
    """
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        t = raw.lower()
        if t in _STOPWORDS:
            continue
        if len(t) <= 2:
            continue
        # crude singular-ish normalisation
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.add(t)
    return tokens


# --------------------------------------------------------------------- #


@dataclass
class ExplainedChunk:
    """One row of the retrieval explanation table.

    Attributes
    ----------
    rank : int
        1-based position in the retrieved list.
    score : float
        Retriever's similarity / rerank score for the chunk.
    chunk_id : str
        Deterministic sha1 id from Phase 3 ingestion.
    source_text : str
        Book title (``Vrikshayurveda`` etc.).
    chapter : str
        Chapter number / ``"full"`` / ``"pages_X_Y"``.
    verse_or_section : str
        Verse marker or ``section_N`` if no verse number was present.
    matched_terms : list[str]
        Tokens present in both the query and the chunk (after
        :func:`tokenize`), sorted for stable display.
    text_with_markers : str
        The chunk text with every matched term wrapped in ``**…**``
        (Markdown-flavoured; the notebook / Streamlit caller can
        convert to its own highlight syntax).
    """

    rank: int
    score: float
    chunk_id: str
    source_text: str
    chapter: str
    verse_or_section: str
    matched_terms: list[str]
    text_with_markers: str


# --------------------------------------------------------------------- #


def _wrap_matched_terms(text: str, matched: set[str]) -> str:
    """Wrap every occurrence of any token in ``matched`` with ``**…**``.

    Case-insensitive, word-boundary-anchored, and tolerant of the same
    crude plural-singular normalisation used by :func:`tokenize` — so
    ``leaves`` in the chunk gets wrapped when the query produced
    ``leaf``.
    """
    if not matched:
        return text

    # Expand each token to "<token>" and "<token>s" to catch the
    # plural form in the raw chunk text. Sort by length descending so
    # the regex prefers the longer match (e.g. ``leaves`` over ``leave``).
    forms: set[str] = set()
    for t in matched:
        forms.add(t)
        forms.add(t + "s")
    pattern = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    if not pattern:
        return text
    rx = re.compile(rf"\b({pattern})\b", flags=re.IGNORECASE)
    return rx.sub(r"**\1**", text)


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #


def explain_chunks(query: str, retrieved_chunks: list[Any]) -> list[ExplainedChunk]:
    """Build one :class:`ExplainedChunk` row per retrieved hit.

    Parameters
    ----------
    query
        The retrieval query string (Strategy A's template, Strategy B's
        LLM rewrite, etc.).
    retrieved_chunks
        List of objects exposing ``chunk_id``, ``text``, ``score``,
        and a ``metadata`` dict with ``source_text`` / ``chapter`` /
        ``verse_or_section`` — i.e. the shape Phase 7's
        :class:`~src.rag.retriever.RetrievedChunk` already provides.
        Plain dicts with the same keys are also accepted.

    Returns
    -------
    list[ExplainedChunk]
        Same order as ``retrieved_chunks``; each row carries its
        ``matched_terms`` (sorted query∩chunk tokens) and a
        ``text_with_markers`` (chunk text with matched terms wrapped
        in ``**…**``).
    """
    q_tokens = tokenize(query or "")
    out: list[ExplainedChunk] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        text = _get(chunk, "text", "")
        chunk_id = _get(chunk, "chunk_id", "")
        score = float(_get(chunk, "score", 0.0))
        meta = _get(chunk, "metadata", {}) or {}
        c_tokens = tokenize(text)
        matched = sorted(q_tokens & c_tokens)
        out.append(
            ExplainedChunk(
                rank=idx,
                score=score,
                chunk_id=str(chunk_id),
                source_text=str(meta.get("source_text", "?")),
                chapter=str(meta.get("chapter", "?")),
                verse_or_section=str(meta.get("verse_or_section", "?")),
                matched_terms=matched,
                text_with_markers=_wrap_matched_terms(text or "", set(matched)),
            )
        )
    return out


# --------------------------------------------------------------------- #
# Back-compat: keep the old highlight_chunks symbol exported so the
# existing tests/explain/test_smoke.py keeps importing cleanly.
# --------------------------------------------------------------------- #


@dataclass
class HighlightedSpan:
    """Character span recording where a chunk supports an answer
    sentence. Phase 9 does not produce these — kept for back-compat
    with the earlier scaffold that the Streamlit demo references."""

    start_char: int
    end_char: int
    answer_sentence_index: int
    similarity: float


@dataclass
class HighlightedChunk:
    """Retrieved chunk + supporting span annotations. Phase 9 emits
    :class:`ExplainedChunk` instead; this dataclass is preserved for
    back-compat with the earlier scaffold."""

    chunk: Any
    spans: list[HighlightedSpan]


def highlight_chunks(answer: str, chunks: list[Any]) -> list[HighlightedChunk]:
    """Back-compat shim around :func:`explain_chunks`.

    The Phase 9 design explains the RETRIEVAL step (query ↔ chunk),
    not the GENERATION step (answer ↔ chunk). For new code, call
    :func:`explain_chunks` directly. This shim returns each input
    chunk wrapped in a :class:`HighlightedChunk` with an empty
    ``spans`` list — enough to keep callers that only checked the
    return shape running.
    """
    del answer  # explicitly unused — see docstring
    return [HighlightedChunk(chunk=c, spans=[]) for c in chunks]


__all__ = [
    "ExplainedChunk",
    "HighlightedChunk",
    "HighlightedSpan",
    "explain_chunks",
    "highlight_chunks",
    "tokenize",
]


def _get(obj: Any, name: str, default: Any) -> Any:
    """Attribute-or-key getter so :func:`explain_chunks` accepts both
    :class:`~src.rag.retriever.RetrievedChunk` and plain dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
