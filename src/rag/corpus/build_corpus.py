"""Phase 3 IKS corpus build orchestrator (entry point).

Reads ``configs/corpus/books.yaml``, processes every ``status: ready``
book end-to-end (OCR → clean → chapter-split → chunk → JSONL → embed
into ChromaDB), and prints a summary plus a manifest JSON.

Run from the repo root::

    python -m src.rag.corpus.build_corpus
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml

from src.rag.corpus.chapter_split import locate_chapters
from src.rag.corpus.chunking import Chunk, chunk_chapter
from src.rag.corpus.cleaning import clean_text
from src.rag.corpus.embed import collection_count, embed_chunks
from src.rag.corpus.ocr import ocr_pdf
from src.utils.logging_setup import get_logger
from src.utils.paths import CONFIGS_DIR, CORPUS_CHUNKS_DIR, PROJECT_ROOT

_LOGGER = get_logger(__name__)

BOOKS_YAML_PATH: Path = CONFIGS_DIR / "corpus" / "books.yaml"
MANIFEST_PATH: Path = CORPUS_CHUNKS_DIR / "_manifest.json"


# --------------------------------------------------------------------- #
# Per-book processing
# --------------------------------------------------------------------- #


def _load_books_config() -> dict[str, Any]:
    with BOOKS_YAML_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_pdf_path(rel: str) -> Path:
    """Resolve a book's PDF path relative to the project root."""
    p = (PROJECT_ROOT / rel).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"PDF not found at {p} (book config said {rel!r})")
    return p


def _ocr_and_clean(book_id: str, pdf_path: Path) -> list[str]:
    """OCR every page of the PDF and return the per-page cleaned text."""
    pages = ocr_pdf(pdf_path, book_id=book_id)
    cleaned = [clean_text(p.raw_text) for p in pages]
    _LOGGER.info(
        "%s: ocr=%d pages, cleaned avg chars/page=%d",
        book_id, len(pages),
        sum(len(p) for p in cleaned) // max(1, len(cleaned)),
    )
    return cleaned


def _chunks_for_full_book(book: dict, cleaned_pages: list[str]) -> list[Chunk]:
    """Chunk a ``scope: full`` book — the whole cleaned text is one block."""
    full_text = "\n\n".join(p for p in cleaned_pages if p.strip())
    meta = {
        "book_id": book["id"],
        "source_text": book["title"],
        "edition": book.get("edition", ""),
        "chapter": "full",
        "original_language": book.get("original_language", "Sanskrit"),
        "translator": book.get("translator", ""),
        "topic_tags": list(book.get("default_topic_tags", [])),
        "metadata_extras": {},
    }
    return chunk_chapter(full_text, meta)


def _chunks_for_scoped_book(book: dict, cleaned_pages: list[str]) -> tuple[list[Chunk], dict[int, int]]:
    """Chunk a ``scope: chapters`` book.

    Returns ``(chunks, found_chapters)`` where ``found_chapters`` is a
    ``{chapter_number: page_span_size}`` map. Chapters that locate_chapters
    couldn't find are omitted (with a WARNING already logged).
    """
    wanted: list[int] = list(book.get("chapters") or [])
    chapter_titles: dict[int, str] = {
        int(k): str(v) for k, v in (book.get("chapter_titles") or {}).items()
    }
    spans = locate_chapters(cleaned_pages, chapter_titles)
    if not spans:
        _LOGGER.warning("%s: locate_chapters found 0 chapters; skipping.", book["id"])
        return [], {}

    chunks: list[Chunk] = []
    found_counts: dict[int, int] = {}
    for chap_num in sorted(wanted):
        span = spans.get(chap_num)
        if span is None:
            continue
        body = "\n\n".join(
            page for page in cleaned_pages[span.start_page_idx:span.end_page_idx] if page.strip()
        )
        if not body.strip():
            _LOGGER.warning(
                "%s ch.%d: located at pages %d..%d but body is empty after cleaning.",
                book["id"], chap_num,
                span.start_page_idx + 1, span.end_page_idx,
            )
            continue
        meta = {
            "book_id": book["id"],
            "source_text": book["title"],
            "edition": book.get("edition", ""),
            "chapter": str(chap_num),
            "original_language": book.get("original_language", "Sanskrit"),
            "translator": book.get("translator", ""),
            "topic_tags": list(book.get("default_topic_tags", [])),
            "metadata_extras": {"chapter_title": span.title},
        }
        chap_chunks = chunk_chapter(body, meta)
        chunks.extend(chap_chunks)
        found_counts[chap_num] = len(chap_chunks)
    return chunks, found_counts


def _chunk_to_jsonl(chunk: Chunk) -> str:
    return json.dumps(asdict(chunk), ensure_ascii=False)


def _write_chunks_jsonl(book_id: str, chunks: list[Chunk]) -> Path:
    CORPUS_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    path = CORPUS_CHUNKS_DIR / f"{book_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(_chunk_to_jsonl(ch) + "\n")
    return path


def _config_hash() -> str:
    return hashlib.sha1(BOOKS_YAML_PATH.read_bytes()).hexdigest()


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def build_corpus() -> dict[str, Any]:
    """Run the full Phase 3 pipeline on every ``status: ready`` book."""
    cfg = _load_books_config()
    books = cfg.get("books", [])
    ready_books = [b for b in books if b.get("status") == "ready"]
    pending_books = [b for b in books if b.get("status") != "ready"]
    if not ready_books:
        _LOGGER.warning("build_corpus: no books with status: ready in %s", BOOKS_YAML_PATH)
        return {}

    summary_rows: list[dict[str, Any]] = []
    total_chunks_embedded = 0
    t0 = time.monotonic()
    for book in ready_books:
        book_id = book["id"]
        _LOGGER.info("=" * 70)
        _LOGGER.info("Processing %s ...", book_id)
        pdf_path = _resolve_pdf_path(book["pdf"])

        cleaned_pages = _ocr_and_clean(book_id, pdf_path)

        chapters_found: dict[int, int]
        if book.get("scope") == "chapters":
            chunks, chapters_found = _chunks_for_scoped_book(book, cleaned_pages)
        else:
            chunks = _chunks_for_full_book(book, cleaned_pages)
            chapters_found = {}

        jsonl_path = _write_chunks_jsonl(book_id, chunks)
        n_embedded = embed_chunks(chunks)
        total_chunks_embedded += n_embedded

        wanted = list(book.get("chapters") or [])
        missing = sorted(set(wanted) - set(chapters_found.keys()))
        summary_rows.append({
            "book_id": book_id,
            "pages_ocr": len(cleaned_pages),
            "scope": book.get("scope", "full"),
            "chapters_wanted": len(wanted) if wanted else None,
            "chapters_found": len(chapters_found) if chapters_found else None,
            "chapters_missing": missing or None,
            "chunks": len(chunks),
            "chunks_embedded": n_embedded,
            "chunks_jsonl": str(jsonl_path.relative_to(PROJECT_ROOT)),
        })

    elapsed = time.monotonic() - t0

    # ---- summary table ----
    print()
    print("=" * 80)
    print(f"Phase 3 corpus build complete in {elapsed:.1f}s")
    print(f"Pending books (not yet processed): {[b['id'] for b in pending_books]}")
    print()
    print(f"{'book':<22} {'pages':>6} {'scope':<10} {'chap':<8} {'chunks':>7} {'embed':>7}")
    print("-" * 80)
    for row in summary_rows:
        chap_field = (
            f"{row['chapters_found']}/{row['chapters_wanted']}"
            if row["chapters_wanted"] is not None
            else "-"
        )
        print(
            f"{row['book_id']:<22} {row['pages_ocr']:>6} {row['scope']:<10} "
            f"{chap_field:<8} {row['chunks']:>7} {row['chunks_embedded']:>7}"
        )
        if row.get("chapters_missing"):
            print(f"  WARN: chapters not found: {row['chapters_missing']}")
    print("-" * 80)
    print(f"Total chunks embedded: {total_chunks_embedded}")
    print(f"ChromaDB collection 'iks_corpus' now holds {collection_count()} vectors")
    print("=" * 80)

    manifest = {
        "config_path": str(BOOKS_YAML_PATH.relative_to(PROJECT_ROOT)),
        "config_sha1": _config_hash(),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed),
        "ready_books": [row["book_id"] for row in summary_rows],
        "pending_books": [b["id"] for b in pending_books],
        "rows": summary_rows,
        "total_chunks_embedded": int(total_chunks_embedded),
        "vector_db_count": int(collection_count()),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _LOGGER.info("Wrote manifest to %s", MANIFEST_PATH)
    return manifest


def main(argv: list[str] | None = None) -> int:
    build_corpus()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
