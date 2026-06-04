"""Verse / passage-level chunking for the IKS corpus.

Produces self-contained chunks of 200–500 tokens (whitespace-split count
as a cheap proxy — close enough for Phase 3's retrieval quality bar),
**never splitting mid-sentence**. We prefer verse boundaries where
detectable (lines that start with a small number like ``"1."``,
``"24.``, or an Arabic-numbered marker mid-line) and fall back to
paragraph boundaries.

Each chunk gets a deterministic ``chunk_id``::

    sha1(book_id || chapter || verse_or_section || first 40 chars of text)

so re-running the pipeline upserts the same rows in ChromaDB and never
creates duplicates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


TARGET_MIN_TOKENS: int = 200
TARGET_MAX_TOKENS: int = 500


# Verse markers: a paragraph that STARTS with one of these forms.
# Examples we want to catch:
#   "1. This is the first verse..."
#   "24. ..."
#   "1-2. ..."
# Sentence-ending periods elsewhere in the line don't qualify.
_VERSE_START_RE = re.compile(r"^\s*(\d{1,3}(?:-\d{1,3})?)[.)]\s+")

# Paragraph break: one or more blank lines.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# Sentence boundary: ., !, ? followed by whitespace and a capital letter
# OR end of string. Good enough for English translations.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=[.!?])\s*$")


@dataclass
class Chunk:
    """One retrievable text chunk + its metadata.

    Attributes mirror the prompt's Locked Decision #6: ``source_text``,
    ``edition``, ``chapter``, ``verse_or_section``, ``topic_tags``,
    ``original_language``, ``translator``, ``chunk_id``. Plus
    ``book_id`` and the chunk text itself.
    """

    chunk_id: str
    book_id: str
    source_text: str           # canonical book title
    edition: str
    chapter: str               # e.g. "55" or "intro" or "section_3"
    verse_or_section: str      # e.g. "1-3" or "section_2"
    topic_tags: list[str]
    original_language: str
    translator: str
    text: str
    metadata_extras: dict[str, Any] = field(default_factory=dict)


def _token_count(text: str) -> int:
    """Whitespace token count — cheap and good enough for chunk-size targeting."""
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_verses(text: str) -> list[tuple[str, str]]:
    """Split into ``(verse_marker, content)`` pairs by verse-start regex.

    If no verse markers are detected, returns ``[("", text)]`` so the
    caller can fall back to paragraph chunking.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    verses: list[tuple[str, str]] = []
    current_marker = ""
    current_buf: list[str] = []
    for para in paragraphs:
        match = _VERSE_START_RE.match(para)
        if match:
            # Flush the previous verse.
            if current_buf:
                verses.append((current_marker, "\n\n".join(current_buf).strip()))
            current_marker = match.group(1)
            stripped = para[match.end():]
            current_buf = [stripped] if stripped else []
        else:
            current_buf.append(para)
    if current_buf:
        verses.append((current_marker, "\n\n".join(current_buf).strip()))

    # If we never hit a verse marker, signal fallback.
    if all(marker == "" for marker, _ in verses):
        return [("", text)]
    return verses


def _safe_split_at_sentence(text: str, soft_limit: int) -> tuple[str, str]:
    """Split ``text`` near ``soft_limit`` tokens, NEVER mid-sentence.

    Returns ``(head, tail)`` where ``head`` ends at the latest sentence
    boundary <= the soft limit. If no sentence boundary exists before
    the limit, returns the whole text as head and an empty tail (i.e.
    we'd rather exceed the soft limit than break a sentence).
    """
    tokens = text.split()
    if len(tokens) <= soft_limit:
        return text, ""

    candidate = " ".join(tokens[:soft_limit])
    # Find the last sentence boundary inside the candidate.
    matches = list(_SENTENCE_BOUNDARY_RE.finditer(candidate))
    if not matches:
        return text, ""
    cut = matches[-1].end()
    head = candidate[:cut].strip()
    if not head:
        return text, ""
    # Rebuild tail by removing the head from the original.
    tail = text[len(head):].lstrip()
    return head, tail


def _chunk_text_into_blocks(text: str) -> list[str]:
    """Pack a passage into 200–500 token blocks, never splitting sentences.

    Strategy: walk paragraph-by-paragraph, accumulating into a buffer
    until adding the next paragraph would overflow ``TARGET_MAX_TOKENS``.
    If the buffer is still below ``TARGET_MIN_TOKENS`` and a single
    paragraph is huge, we sentence-split the paragraph.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    blocks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def _flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            blocks.append("\n\n".join(buf).strip())
            buf = []
            buf_tokens = 0

    for para in paragraphs:
        para_tokens = _token_count(para)

        # Paragraph is enormous on its own — sentence-split into pieces.
        if para_tokens > TARGET_MAX_TOKENS:
            _flush()
            remaining = para
            while _token_count(remaining) > TARGET_MAX_TOKENS:
                head, tail = _safe_split_at_sentence(remaining, TARGET_MAX_TOKENS)
                if not head or head == remaining:
                    # No sentence boundary found — accept oversize block
                    # rather than splitting mid-sentence.
                    blocks.append(remaining.strip())
                    remaining = ""
                    break
                blocks.append(head)
                remaining = tail
            if remaining.strip():
                buf = [remaining.strip()]
                buf_tokens = _token_count(remaining)
            continue

        # Otherwise: pack into buf, flushing when min-target is met and
        # adding the next para would overshoot the max.
        if buf_tokens + para_tokens > TARGET_MAX_TOKENS and buf_tokens >= TARGET_MIN_TOKENS:
            _flush()
        buf.append(para)
        buf_tokens += para_tokens

    _flush()
    return [b for b in blocks if b.strip()]


def _compute_chunk_id(
    book_id: str, chapter: str, verse_or_section: str, text: str,
) -> str:
    head = (text or "")[:40]
    payload = f"{book_id}|{chapter}|{verse_or_section}|{head}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def chunk_chapter(
    text: str,
    meta: dict[str, Any],
) -> list[Chunk]:
    """Chunk one chapter's cleaned English text.

    Parameters
    ----------
    text
        Cleaned text for the chapter.
    meta
        Must contain keys: ``book_id``, ``source_text``, ``edition``,
        ``chapter`` (string), ``original_language``, ``translator``,
        ``topic_tags`` (list[str]). May contain ``metadata_extras``.

    Returns
    -------
    list[Chunk]
        Self-contained, deterministic chunks. Empty list if the input
        text has no usable content.
    """
    if not text or not text.strip():
        return []

    book_id = meta["book_id"]
    chapter = str(meta.get("chapter", ""))
    source_text = meta["source_text"]
    edition = meta["edition"]
    original_language = meta["original_language"]
    translator = meta["translator"]
    topic_tags = list(meta.get("topic_tags", []))
    metadata_extras = dict(meta.get("metadata_extras", {}))

    verses = _split_verses(text)
    chunks: list[Chunk] = []

    if len(verses) == 1 and verses[0][0] == "":
        # No verse markers — paragraph-pack the whole chapter.
        blocks = _chunk_text_into_blocks(text)
        for i, block in enumerate(blocks, 1):
            vs = f"section_{i}"
            chunks.append(
                Chunk(
                    chunk_id=_compute_chunk_id(book_id, chapter, vs, block),
                    book_id=book_id,
                    source_text=source_text,
                    edition=edition,
                    chapter=chapter,
                    verse_or_section=vs,
                    topic_tags=topic_tags,
                    original_language=original_language,
                    translator=translator,
                    text=block,
                    metadata_extras=metadata_extras,
                )
            )
        return chunks

    # Verse-aware packing: group consecutive verses up to TARGET_MAX_TOKENS,
    # while keeping each chunk in the verse range. Single huge verses get
    # sentence-split.
    buf_verses: list[tuple[str, str]] = []
    buf_tokens = 0

    def _emit(buf: list[tuple[str, str]]) -> None:
        if not buf:
            return
        first_marker = buf[0][0] or "1"
        last_marker = buf[-1][0] or first_marker
        marker = first_marker if first_marker == last_marker else f"{first_marker}-{last_marker}"
        body = "\n\n".join(
            (f"{m}. {body}" if m else body) for m, body in buf if body
        ).strip()
        if not body:
            return
        chunks.append(
            Chunk(
                chunk_id=_compute_chunk_id(book_id, chapter, marker, body),
                book_id=book_id,
                source_text=source_text,
                edition=edition,
                chapter=chapter,
                verse_or_section=marker,
                topic_tags=topic_tags,
                original_language=original_language,
                translator=translator,
                text=body,
                metadata_extras=metadata_extras,
            )
        )

    for marker, content in verses:
        verse_tokens = _token_count(content)
        if verse_tokens > TARGET_MAX_TOKENS:
            # Emit anything buffered, then split this oversized verse.
            _emit(buf_verses); buf_verses = []; buf_tokens = 0
            sub_blocks = _chunk_text_into_blocks(content)
            for sub_idx, sub_text in enumerate(sub_blocks, 1):
                sub_marker = f"{marker}.{sub_idx}" if marker else f"section_{sub_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=_compute_chunk_id(book_id, chapter, sub_marker, sub_text),
                        book_id=book_id,
                        source_text=source_text,
                        edition=edition,
                        chapter=chapter,
                        verse_or_section=sub_marker,
                        topic_tags=topic_tags,
                        original_language=original_language,
                        translator=translator,
                        text=sub_text,
                        metadata_extras=metadata_extras,
                    )
                )
            continue

        if buf_tokens + verse_tokens > TARGET_MAX_TOKENS and buf_tokens >= TARGET_MIN_TOKENS:
            _emit(buf_verses); buf_verses = []; buf_tokens = 0
        buf_verses.append((marker, content))
        buf_tokens += verse_tokens

    _emit(buf_verses)
    return chunks


__all__ = [
    "Chunk",
    "TARGET_MAX_TOKENS",
    "TARGET_MIN_TOKENS",
    "chunk_chapter",
]
