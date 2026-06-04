"""Phase 3 IKS corpus pipeline (master plan §15).

Sub-package containing the OCR → clean → chapter-split → chunk →
embed → ChromaDB pipeline used to ingest classical agri-Sanskrit texts.
The pipeline is **config-driven** via ``configs/corpus/books.yaml`` —
adding a new book to the corpus is one YAML entry, never a code change.

Run::

    python -m src.rag.corpus.build_corpus
    python -m src.rag.corpus.query_smoke

The pre-existing modules in :mod:`src.rag` (``chunker``, ``embedder``,
``retriever``, etc.) are Phase 7 stubs that handle the retrieval-side
of the RAG stack; Phase 3 sits below them and produces the ChromaDB
collection they consume.
"""

from src.rag.corpus.chapter_split import locate_chapters
from src.rag.corpus.chunking import Chunk, chunk_chapter
from src.rag.corpus.cleaning import clean_text, drop_devanagari
from src.rag.corpus.ocr import PageText, ocr_pdf

__all__ = [
    "Chunk",
    "PageText",
    "chunk_chapter",
    "clean_text",
    "drop_devanagari",
    "locate_chapters",
    "ocr_pdf",
]
