"""Embed chunks with BAAI/bge-large-en-v1.5 and upsert into ChromaDB.

Per Locked Decisions #7 and #8:

- Embedding model: ``BAAI/bge-large-en-v1.5`` (1024-dim) via
  ``sentence-transformers``. Runs on GPU when available, falls back to
  CPU (slow but workable for a few thousand chunks).
- Store: ChromaDB ``PersistentClient`` at
  ``corpus/vector_db/``. Collection name: ``iks_corpus``.
- ``upsert`` keyed by the chunk's deterministic ``chunk_id`` so
  re-running the pipeline never creates duplicates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.rag.corpus.chunking import Chunk
from src.utils.logging_setup import get_logger
from src.utils.paths import VECTOR_DB_DIR

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_LOGGER = get_logger(__name__)

EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM: int = 1024
COLLECTION_NAME: str = "iks_corpus"
DEFAULT_BATCH_SIZE: int = 16


def _load_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOGGER.info("Loading embedding model %s on %s ...", EMBEDDING_MODEL_NAME, device)
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def _get_collection(client_factory=None):
    """Return the ``iks_corpus`` Chroma collection, creating it if needed."""
    if client_factory is None:
        import chromadb  # noqa: PLC0415
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    else:
        client = client_factory()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _chunk_to_metadata(chunk: Chunk) -> dict[str, str]:
    """Flatten the chunk's metadata for ChromaDB (it accepts str/int/float/bool only)."""
    return {
        "book_id": chunk.book_id,
        "source_text": chunk.source_text,
        "edition": chunk.edition,
        "chapter": chunk.chapter,
        "verse_or_section": chunk.verse_or_section,
        "topic_tags": ", ".join(chunk.topic_tags),
        "original_language": chunk.original_language,
        "translator": chunk.translator,
    }


def embed_chunks(
    chunks: Sequence[Chunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed every chunk and upsert into the ``iks_corpus`` collection.

    Returns the number of vectors written (== ``len(chunks)`` on success).
    """
    if not chunks:
        _LOGGER.info("embed_chunks: no chunks to embed.")
        return 0

    model = _load_model()
    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    for ch in chunks:
        ids.append(ch.chunk_id)
        documents.append(ch.text)
        metadatas.append(_chunk_to_metadata(ch))

    total = len(ids)
    _LOGGER.info("Embedding %d chunks in batches of %d ...", total, batch_size)
    written = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]
        embeddings = model.encode(
            batch_docs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings.tolist(),
        )
        written += len(batch_ids)
        _LOGGER.info(
            "  ... upserted %d/%d (batch %d-%d)",
            written, total, start + 1, end,
        )

    _LOGGER.info("embed_chunks: %d vectors upserted into %s.", written, COLLECTION_NAME)
    return written


def collection_count() -> int:
    """Number of vectors currently in the ``iks_corpus`` collection."""
    return _get_collection().count()


__all__ = [
    "COLLECTION_NAME",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_NAME",
    "collection_count",
    "embed_chunks",
]
