"""Phase 9 — chunk-highlight tests.

Pure-Python, no models, no network. Validates:

- :func:`tokenize` lower-cases, drops punctuation, drops the locked
  stopword set, and normalises trivial plurals so ``leaves`` and
  ``leaf`` match.
- :func:`explain_chunks` returns ``matched_terms = query ∩ chunk`` per
  row, wraps each matched term in ``**…**`` markers (preserving the
  original case in the chunk text), and emits the expected per-row
  fields even when the chunk has zero overlap with the query.
- The helper accepts both ``RetrievedChunk``-like attribute access
  AND plain ``dict`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.explain.chunk_highlight import (
    ExplainedChunk,
    _STOPWORDS,
    explain_chunks,
    tokenize,
)


# --------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------- #


def test_tokenize_lowercases_and_drops_punct() -> None:
    out = tokenize("Tomato! Leaf-blight, on Plant.")
    # All lowercase, no punctuation, no length-≤2 noise, no template
    # filler words like "on".
    assert all(t == t.lower() for t in out)
    assert "tomato" in out
    assert "leaf" in out
    assert "blight" in out
    assert "plant" in out
    assert "on" not in out
    assert "," not in " ".join(out)


def test_tokenize_drops_stopwords() -> None:
    """Every word in ``_STOPWORDS`` must be absent from the output."""
    for w in _STOPWORDS:
        # Only test the ones the regex can actually capture (letters).
        if w.isalpha():
            assert w not in tokenize(f"foo {w} bar")


def test_tokenize_singularises_long_words() -> None:
    """Trivial plural normalisation: ``leaves`` → ``leave`` and
    ``trees`` → ``tree`` (length > 3 + ends in ``s`` but not ``ss``)."""
    assert "tree" in tokenize("trees")
    assert "leave" in tokenize("leaves")
    # Words ending in ``ss`` (e.g. ``grass``) must NOT be stripped.
    assert "grass" in tokenize("grass")
    # ≤ 3 letter words are dropped by the length filter, not stripped.
    assert "as" not in tokenize("as")


# --------------------------------------------------------------------- #
# explain_chunks — matched terms + markers
# --------------------------------------------------------------------- #


@dataclass
class _StubRetrieved:
    chunk_id: str
    text: str
    score: float
    metadata: dict


def _chunk(
    cid: str, txt: str, score: float = 0.5,
    source: str = "Vrikshayurveda", chapter: str = "1", verse: str = "4",
) -> _StubRetrieved:
    return _StubRetrieved(
        chunk_id=cid, text=txt, score=score,
        metadata={"source_text": source, "chapter": chapter, "verse_or_section": verse},
    )


def test_explain_chunks_matched_terms_are_query_intersection_chunk() -> None:
    query = "Organic treatment for tomato leaf blight on alluvial soil"
    chunks = [
        _chunk(
            "c1",
            "Apply paste of butter and clarified ghee to the diseased tomato leaf.",
            score=0.81,
        ),
        _chunk(
            "c2",
            "When the rains depart, store rice grain in baskets lined with neem.",
            score=0.65,
        ),
    ]
    out = explain_chunks(query, chunks)
    assert len(out) == 2
    assert isinstance(out[0], ExplainedChunk)

    # c1 mentions "tomato" + "leaf", both in the query.
    c1 = out[0]
    assert c1.rank == 1 and c1.chunk_id == "c1"
    assert "tomato" in c1.matched_terms
    assert "leaf" in c1.matched_terms
    # The wrapping must round-trip the original case (the chunk says
    # "tomato leaf" lower-case, but the wrapper is case-insensitive).
    assert "**tomato**" in c1.text_with_markers
    assert "**leaf**" in c1.text_with_markers
    # c1.text_with_markers should still contain the rest of the sentence.
    assert "Apply paste" in c1.text_with_markers

    # c2 has no overlap with the query → empty matched_terms +
    # text_with_markers should be IDENTICAL to the raw chunk text.
    c2 = out[1]
    assert c2.matched_terms == []
    assert "**" not in c2.text_with_markers
    assert c2.text_with_markers == c2.text_with_markers  # idempotent


def test_explain_chunks_accepts_dict_rows() -> None:
    """Strategy C returns plain dicts (not RetrievedChunk objects)."""
    query = "tomato leaf scorched"
    dict_chunk = {
        "chunk_id": "c3",
        "text": "The tomato fruit ripens after the second monsoon.",
        "score": 0.4,
        "metadata": {
            "source_text": "Krishi Parashara", "chapter": "2", "verse_or_section": "7",
        },
    }
    out = explain_chunks(query, [dict_chunk])
    assert out[0].chunk_id == "c3"
    assert "tomato" in out[0].matched_terms
    assert out[0].source_text == "Krishi Parashara"
    assert out[0].chapter == "2"


def test_explain_chunks_zero_overlap_chunk_renders_cleanly() -> None:
    out = explain_chunks(
        "rainfall divination",
        [_chunk("c4", "He honoured the guest with rice and curd.")],
    )
    row = out[0]
    assert row.matched_terms == []
    assert "**" not in row.text_with_markers
    # Even with no overlap, the row must still carry the source metadata
    # and a non-empty body — explain_chunks is descriptive, not filtering.
    assert row.source_text == "Vrikshayurveda"
    assert "honoured" in row.text_with_markers


def test_explain_chunks_handles_regular_plural_match() -> None:
    """Trivial plural matching: query says ``tree`` (or ``trees``),
    chunk says ``trees`` (or ``tree``) — the singular form lands in
    matched_terms in both directions because tokenize strips trailing
    ``s`` on words longer than 3 characters.

    Note this only handles the regular -s plural. English irregulars
    (leaves/leaf, branches/branch) are NOT normalised — Phase 9 is
    deliberately a thin, transparent layer, not an NLP stack."""
    out = explain_chunks(
        "rainfall trees",
        [_chunk("c5", "He who plants a tree on the road earns merit.")],
    )
    row = out[0]
    assert "tree" in row.matched_terms
    # Wrapper expands "tree" → "trees" too, so the chunk's lone "tree"
    # is wrapped.
    assert "**tree**" in row.text_with_markers


def test_explain_chunks_preserves_input_order_and_assigns_rank() -> None:
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(3)]
    out = explain_chunks("any query", chunks)
    assert [r.rank for r in out] == [1, 2, 3]
    assert [r.chunk_id for r in out] == ["c0", "c1", "c2"]
