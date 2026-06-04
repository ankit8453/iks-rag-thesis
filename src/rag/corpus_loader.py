"""Phase 7 RAG corpus loader — HF dataset → ChromaDB.

Phase 3 built the corpus on a Windows laptop; Phase 7 runs on a Colab /
Linux GPU runtime. Rather than ship the laptop's ``corpus/vector_db/``
(binary, ChromaDB-version-sensitive), we publish the chunks as a
**private** HF dataset (``ankit-iiitdmj/iks-corpus-chunks``) and
re-embed them in the Colab session. This module provides the two
functions that bridge the gap:

- :func:`load_chunks_from_hf` pulls the dataset and returns a list of
  plain ``dict`` rows with the full Phase 3 metadata schema.
- :func:`build_chroma` re-embeds those rows with BAAI/bge-large-en-v1.5
  and upserts them into a ChromaDB ``PersistentClient`` collection
  keyed by ``chunk_id`` (deterministic sha1 — idempotent).

Single-process design is fine on Linux: the Windows-only torch + grpc
DLL conflict (see ``feedback_chromadb_torch_windows_dll.md``) does NOT
manifest in the Colab runtime.

This replaces the Phase-4 ``IKSCorpus`` stub that previously lived here;
it raised ``NotImplementedError`` and nothing in the live codebase
consumed it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


DEFAULT_CHUNKS_REPO: str = "ankit-iiitdmj/iks-corpus-chunks"
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
DEFAULT_COLLECTION_NAME: str = "iks_corpus"

REQUIRED_FIELDS: tuple[str, ...] = (
    "book_id",
    "chunk_id",
    "source_text",
    "edition",
    "chapter",
    "verse_or_section",
    "topic_tags",
    "original_language",
    "translator",
    "text",
)


def load_chunks_from_hf(
    repo: str = DEFAULT_CHUNKS_REPO,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Pull the private chunks dataset and return a list of metadata-rich rows.

    Parameters
    ----------
    repo
        HF dataset repo id (default ``ankit-iiitdmj/iks-corpus-chunks``).
    split
        Dataset split to load. The Phase 7 upload uses a single
        ``train`` split.

    Returns
    -------
    list[dict]
        One dict per chunk, keys covering the full Phase 3 schema (see
        :data:`REQUIRED_FIELDS`).
    """
    from datasets import load_dataset  # noqa: PLC0415

    _LOGGER.info("Loading chunks from %s (split=%s) ...", repo, split)
    ds = load_dataset(repo, split=split)

    rows: list[dict[str, Any]] = []
    for row in ds:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise RuntimeError(
                f"Row missing required field(s) {missing} in {repo}: "
                f"present keys = {sorted(row)}"
            )
        rows.append({k: row[k] for k in REQUIRED_FIELDS})
    _LOGGER.info("Loaded %d chunks from %s.", len(rows), repo)
    return rows


def _embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
):
    """Encode ``texts`` with sentence-transformers — float32 numpy matrix."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOGGER.info("Loading embedding model %s on %s ...", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    _LOGGER.info("Embedding %d texts in batches of %d ...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def build_chroma(
    chunks: list[dict[str, Any]],
    persist_dir: str | Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
):
    """Re-embed ``chunks`` and upsert into a ChromaDB ``PersistentClient`` collection.

    Idempotent: upserts by ``chunk_id``. Re-running with the same chunks
    overwrites the same rows in place.

    Returns
    -------
    chromadb.api.models.Collection.Collection
        The populated collection, ready for :class:`HybridRetriever`.
    """
    import chromadb  # noqa: PLC0415

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(name=collection_name)

    if not chunks:
        _LOGGER.warning("build_chroma: no chunks to embed; returning empty collection.")
        return collection

    texts = [c["text"] for c in chunks]
    embeddings = _embed_texts(
        texts, model_name=embedding_model, batch_size=batch_size,
    )

    ids = [c["chunk_id"] for c in chunks]
    metadatas: list[dict[str, str]] = []
    for c in chunks:
        # ChromaDB requires metadata values be str / int / float / bool;
        # stringify everything (matches Phase 3's build_corpus output).
        topic_tags = c.get("topic_tags", "")
        if isinstance(topic_tags, list):
            topic_tags = ", ".join(str(t) for t in topic_tags)
        metadatas.append(
            {
                "book_id": str(c["book_id"]),
                "source_text": str(c["source_text"]),
                "edition": str(c["edition"]),
                "chapter": str(c["chapter"]),
                "verse_or_section": str(c["verse_or_section"]),
                "topic_tags": str(topic_tags),
                "original_language": str(c["original_language"]),
                "translator": str(c["translator"]),
            }
        )

    total = len(ids)
    _LOGGER.info("Upserting %d chunks into Chroma collection %r ...", total, collection_name)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=[e.tolist() for e in embeddings[start:end]],
        )
        _LOGGER.info("  ... upserted %d/%d", end, total)

    return collection


__all__ = [
    "DEFAULT_CHUNKS_REPO",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "REQUIRED_FIELDS",
    "build_chroma",
    "load_chunks_from_hf",
]
