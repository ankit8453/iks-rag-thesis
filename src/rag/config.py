"""Pydantic schema for the RAG pipeline.

Per master reference §12 (RAG): primary embeddings are BAAI/bge-large-en,
generator is Llama-3.1-8B in 4-bit, retrieval is dense + BM25 hybrid with a
cross-encoder rerank. No LangChain / LlamaIndex (ADR-0003).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.utils.config import BaseConfig


class ChunkerConfig(BaseConfig):
    """Chunking parameters shared by sentence-window and sliding-window strategies."""

    strategy: Literal["paragraph", "sliding_window"] = "paragraph"
    target_tokens: int = Field(256, ge=32, le=2048)
    overlap_tokens: int = Field(32, ge=0, le=512)


class RAGConfig(BaseConfig):
    """Top-level RAG pipeline configuration."""

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    multilingual_fallback: str = "BAAI/bge-m3"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"

    # Retrieval depth
    top_k_dense: int = Field(20, ge=1)
    top_k_bm25: int = Field(20, ge=1)
    top_k_rerank: int = Field(5, ge=1)

    # Vector store
    vector_store: Literal["chromadb", "faiss"] = "chromadb"
    collection_name: str = "iks_corpus"

    # Generator (LLM)
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_quantization: Literal["none", "4bit", "8bit"] = "4bit"
    llm_max_new_tokens: int = Field(512, ge=16, le=4096)
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # Hybrid fusion
    hybrid_alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Weight on dense score in dense+BM25 fusion (1.0 = dense only).",
    )

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    seed: int = Field(42, ge=0)
