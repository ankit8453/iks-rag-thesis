"""Tests for :mod:`src.rag.corpus.chunking`."""

from __future__ import annotations

import pytest

from src.rag.corpus.chunking import (
    TARGET_MAX_TOKENS,
    TARGET_MIN_TOKENS,
    chunk_chapter,
)


def _make_meta(**overrides) -> dict:
    base = {
        "book_id": "test_book",
        "source_text": "Test Book",
        "edition": "ed.1",
        "chapter": "55",
        "original_language": "Sanskrit",
        "translator": "Tester",
        "topic_tags": ["pest_control"],
    }
    base.update(overrides)
    return base


def _make_long_text(paragraphs: int = 20, words_per_paragraph: int = 20) -> str:
    """Plain prose without verse markers — exercises the paragraph-packing path."""
    seed = "Apply paste of butter and clarified ghee to the roots of the tree."
    para = " ".join([seed] * (words_per_paragraph // 12 + 1))
    para = " ".join(para.split()[:words_per_paragraph]) + "."
    return ("\n\n".join([para] * paragraphs)).strip()


def _make_verse_text(n_verses: int = 5, words_per_verse: int = 25) -> str:
    body = " ".join(["the seed is sown in moist ground."] * (words_per_verse // 6 + 1))
    body = " ".join(body.split()[:words_per_verse])
    parts = [f"{i}. {body}" for i in range(1, n_verses + 1)]
    return "\n\n".join(parts)


def test_chunk_chapter_returns_no_chunks_for_empty_input() -> None:
    assert chunk_chapter("", _make_meta()) == []
    assert chunk_chapter("   \n  \n   ", _make_meta()) == []


def test_chunks_have_all_metadata_fields_populated() -> None:
    chunks = chunk_chapter(_make_long_text(paragraphs=30), _make_meta())
    assert chunks
    for ch in chunks:
        assert ch.book_id == "test_book"
        assert ch.source_text == "Test Book"
        assert ch.edition == "ed.1"
        assert ch.chapter == "55"
        assert ch.original_language == "Sanskrit"
        assert ch.translator == "Tester"
        assert ch.topic_tags == ["pest_control"]
        assert ch.verse_or_section != ""
        assert ch.chunk_id and len(ch.chunk_id) == 40   # sha1 hex
        assert ch.text.strip() != ""


def test_chunks_respect_token_bounds_except_when_single_paragraph_is_huge() -> None:
    text = _make_long_text(paragraphs=40, words_per_paragraph=50)  # 2000 words
    chunks = chunk_chapter(text, _make_meta())
    # All inner chunks should fit under TARGET_MAX_TOKENS; the LAST chunk
    # may be small if the tail is short.
    for ch in chunks[:-1]:
        n = len(ch.text.split())
        assert n <= TARGET_MAX_TOKENS, f"Chunk overflowed: {n} > {TARGET_MAX_TOKENS}"
    # The total text isn't lost.
    rejoined = " ".join(ch.text for ch in chunks)
    assert "Apply paste of butter" in rejoined


def test_chunk_ids_are_unique_and_stable_across_runs() -> None:
    text = _make_verse_text(n_verses=20, words_per_verse=30)
    run_a = chunk_chapter(text, _make_meta())
    run_b = chunk_chapter(text, _make_meta())
    ids_a = [c.chunk_id for c in run_a]
    ids_b = [c.chunk_id for c in run_b]
    # Stable across runs (deterministic sha1).
    assert ids_a == ids_b
    # Within one run, ids are unique.
    assert len(ids_a) == len(set(ids_a))


def test_chunk_chapter_prefers_verse_boundaries_when_available() -> None:
    text = _make_verse_text(n_verses=3, words_per_verse=15)
    chunks = chunk_chapter(text, _make_meta())
    # At least one chunk's verse_or_section should mention a numbered verse
    # (single "1" or a range like "1-3"). The fallback "section_N" form
    # is the no-verse-markers path; here we DO have markers.
    has_verse_marker = any(
        ch.verse_or_section and not ch.verse_or_section.startswith("section_")
        for ch in chunks
    )
    assert has_verse_marker, (
        f"Verse-marker text should produce verse-labelled chunks, got: "
        f"{[c.verse_or_section for c in chunks]}"
    )


def test_no_chunk_text_is_empty() -> None:
    text = _make_verse_text(n_verses=15, words_per_verse=20)
    chunks = chunk_chapter(text, _make_meta())
    for ch in chunks:
        assert ch.text.strip() != ""
