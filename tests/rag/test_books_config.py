"""Phase 3b — books.yaml registry shape + ready_external skip behaviour.

Pure config / loader tests: no PDFs touched, no OCR, no embeddings.

Guards three invariants Phase 3b depends on:

1. ``configs/corpus/books.yaml`` parses, and every entry has the minimum
   fields the loader inspects (``id``, ``status``).
2. The two Phase-3b external-OCR books (``krishi_parashara`` and
   ``upavanavinoda``) carry the new contract correctly:
   ``status: ready_external``, ``ocr_method: gemini_external``,
   ``scope: pages``, a two-element ``page_range``, and a
   ``text_source`` path pointing under ``corpus/ocr_external/``.
3. When the ``text_source`` file does NOT exist yet, the loader skip
   path in ``_chunks_for_external_book`` returns ``None`` (and logs a
   clear "awaiting" message) — it does not raise. This is the
   invariant the build pipeline relies on to skip gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.rag.corpus.build_corpus import _chunks_for_external_book
from src.utils.paths import PROJECT_ROOT

BOOKS_YAML = PROJECT_ROOT / "configs" / "corpus" / "books.yaml"
EXTERNAL_BOOK_IDS = {"krishi_parashara", "upavanavinoda"}


def _load_books() -> list[dict]:
    raw = yaml.safe_load(BOOKS_YAML.read_text(encoding="utf-8"))
    assert isinstance(raw, dict) and "books" in raw, (
        f"{BOOKS_YAML} must be a mapping with a top-level 'books:' list"
    )
    books = raw["books"]
    assert isinstance(books, list) and books, "books: must be a non-empty list"
    return books


def test_books_yaml_parses_and_has_required_fields() -> None:
    books = _load_books()
    seen_ids: set[str] = set()
    for b in books:
        assert "id" in b, f"book entry missing id: {b}"
        assert "status" in b, f"book entry missing status: {b}"
        assert b["id"] not in seen_ids, f"duplicate book id: {b['id']}"
        seen_ids.add(b["id"])


def test_external_books_have_phase3b_contract() -> None:
    books = {b["id"]: b for b in _load_books()}
    missing = EXTERNAL_BOOK_IDS - set(books)
    assert not missing, f"books.yaml is missing Phase 3b entries: {missing}"

    for bid in EXTERNAL_BOOK_IDS:
        b = books[bid]
        assert b.get("status") == "ready_external", (
            f"{bid}: status must be 'ready_external', got {b.get('status')!r}"
        )
        assert b.get("ocr_method") == "gemini_external", (
            f"{bid}: ocr_method must be 'gemini_external', got {b.get('ocr_method')!r}"
        )
        assert b.get("scope") == "pages", (
            f"{bid}: scope must be 'pages', got {b.get('scope')!r}"
        )

        pr = b.get("page_range")
        assert isinstance(pr, list) and len(pr) == 2, (
            f"{bid}: page_range must be a 2-element list, got {pr!r}"
        )
        assert all(isinstance(x, int) for x in pr), (
            f"{bid}: page_range entries must be ints, got {pr!r}"
        )
        assert pr[0] < pr[1], f"{bid}: page_range start must precede end, got {pr!r}"

        ts = b.get("text_source")
        assert isinstance(ts, str) and ts, f"{bid}: text_source must be a non-empty str"
        assert ts.startswith("corpus/ocr_external/"), (
            f"{bid}: text_source must live under corpus/ocr_external/, got {ts!r}"
        )
        assert ts.endswith(".md"), f"{bid}: text_source must be a .md file, got {ts!r}"


def test_external_loader_skips_when_text_source_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_chunks_for_external_book` returns None (no raise) when the
    text_source markdown file is not yet present."""
    fake_book = {
        "id": "krishi_parashara",
        "title": "Krishi Parashara",
        "edition": "Bibliotheca Indica, 1960",
        "original_language": "Sanskrit",
        "translator": "Majumdar",
        "ocr_method": "gemini_external",
        "scope": "pages",
        "page_range": [94, 119],
        "text_source": "corpus/ocr_external/__does_not_exist__.md",
        "default_topic_tags": ["rainfall"],
    }
    with caplog.at_level("WARNING"):
        chunks = _chunks_for_external_book(fake_book)
    assert chunks is None
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "awaiting" in msgs.lower() or "skipping" in msgs.lower(), (
        f"expected an 'awaiting Gemini OCR ... skipping' log, got: {msgs!r}"
    )


def test_external_loader_skips_when_text_source_missing_from_field() -> None:
    """No text_source at all => skip, not raise."""
    fake_book = {"id": "krishi_parashara", "ocr_method": "gemini_external"}
    chunks = _chunks_for_external_book(fake_book)
    assert chunks is None


def test_external_loader_chunks_when_text_source_present(tmp_path: Path) -> None:
    """When the text_source exists and has content, the loader returns
    a non-empty chunk list with the expected metadata threading."""
    # Build a fake text_source under PROJECT_ROOT so the function's
    # relative-path resolution succeeds.
    rel = Path("corpus/ocr_external/_pytest_dummy.md")
    abs_path = PROJECT_ROOT / rel
    abs_path.write_text(
        "1. The first verse of the test fixture.\n\n"
        "2. The second verse of the test fixture.\n",
        encoding="utf-8",
    )
    try:
        fake_book = {
            "id": "krishi_parashara",
            "title": "Krishi Parashara",
            "edition": "Bibliotheca Indica, 1960",
            "original_language": "Sanskrit",
            "translator": "Majumdar",
            "ocr_method": "gemini_external",
            "scope": "pages",
            "page_range": [94, 119],
            "text_source": str(rel).replace("\\", "/"),
            "default_topic_tags": ["rainfall"],
        }
        chunks = _chunks_for_external_book(fake_book)
        assert chunks is not None and len(chunks) >= 1
        first = chunks[0]
        # Chunk is a flat dataclass — fields are attributes, not a dict.
        assert first.book_id == "krishi_parashara"
        assert first.chapter == "pages_94_119"
        assert first.source_text == "Krishi Parashara"
        assert first.original_language == "Sanskrit"
        assert first.translator == "Majumdar"
        assert "rainfall" in first.topic_tags
        extras = first.metadata_extras or {}
        assert extras.get("ocr_method") == "gemini_external"
        assert extras.get("page_range") == [94, 119]
    finally:
        if abs_path.exists():
            abs_path.unlink()
