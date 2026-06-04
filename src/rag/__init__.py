"""Retrieval-Augmented Generation pipeline over IKS corpora.

Implements contributions C1 (corpus), C3 (faithfulness-aware RAG), C4
(hallucination measurement) and C5 (cause-conditional retrieval). No
LangChain or LlamaIndex (ADR-0003); the orchestration is plain Python
wrapping HuggingFace + chromadb + rank-bm25.

Phase 7 public surface:

- :func:`load_chunks_from_hf` — pull the private
  ``ankit-iiitdmj/iks-corpus-chunks`` dataset into a list of metadata-
  rich rows.
- :func:`build_chroma` — re-embed those rows with BGE-large and upsert
  into a ChromaDB collection.
- :class:`HybridRetriever` — dense + BM25 + cross-encoder rerank, all
  stages toggleable for Phase 11 §27 ablations.
- :class:`RetrievedChunk` — retrieval result dataclass (also consumed
  by :mod:`src.explain.chunk_highlight`).
- :class:`GroundedGenerator` — Llama-3.1-8B 4-bit + master plan §17
  grounded-advisor system prompt.
- :class:`RAGPipeline` — top-level orchestrator with a model-agnostic
  generator seam (the Phase 8 multimodal-context plug-in point).

The Phase-3 sub-package :mod:`src.rag.corpus` (OCR/clean/chapter-split/
chunk/embed/build_corpus/query_smoke) is unrelated to the retrieval
pipeline here — that one builds the corpus on the laptop; this one
consumes it on Colab.
"""

from src.rag.chunker import Chunk, Chunker
from src.rag.config import ChunkerConfig, RAGConfig
from src.rag.corpus_loader import (
    DEFAULT_CHUNKS_REPO,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    build_chroma,
    load_chunks_from_hf,
)
from src.rag.embedder import Embedder
from src.rag.generator import (
    DEFAULT_MODEL_NAME,
    GenerationResult,
    GroundedGenerator,
    LlamaGenerator,
    SYSTEM_PROMPT_V17,
    extract_citations,
)
from src.rag.pipeline import RAGAnswer, RAGPipeline
from src.rag.prompts import RAG_PROMPT_TEMPLATE
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import (
    BM25Retriever,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOP_K_DENSE,
    DEFAULT_TOP_K_RERANK,
    DEFAULT_TOP_K_SPARSE,
    DenseRetriever,
    HybridRetriever,
    RRF_K,
    RetrievedChunk,
)

__all__ = [
    "BM25Retriever",
    "Chunk",
    "Chunker",
    "ChunkerConfig",
    "CrossEncoderReranker",
    "DEFAULT_CHUNKS_REPO",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_TOP_K_DENSE",
    "DEFAULT_TOP_K_RERANK",
    "DEFAULT_TOP_K_SPARSE",
    "DenseRetriever",
    "Embedder",
    "GenerationResult",
    "GroundedGenerator",
    "HybridRetriever",
    "LlamaGenerator",
    "RAGAnswer",
    "RAGConfig",
    "RAGPipeline",
    "RAG_PROMPT_TEMPLATE",
    "RRF_K",
    "RetrievedChunk",
    "SYSTEM_PROMPT_V17",
    "build_chroma",
    "extract_citations",
    "load_chunks_from_hf",
]
