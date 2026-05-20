"""Smoke tests for the RAG module."""

from __future__ import annotations

import pytest

from src.rag import (
    RAG_PROMPT_TEMPLATE,
    BM25Retriever,
    Chunker,
    CrossEncoderReranker,
    DenseRetriever,
    Embedder,
    HybridRetriever,
    LlamaGenerator,
    RAGConfig,
)
from src.rag.config import ChunkerConfig


def test_rag_config_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.reranker_model == "BAAI/bge-reranker-base"
    assert cfg.llm_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.llm_quantization == "4bit"
    assert cfg.vector_store == "chromadb"


def test_chunker_instantiation() -> None:
    cfg = ChunkerConfig()
    chunker = Chunker(cfg)
    assert chunker.config.strategy == "paragraph"


def test_retriever_class_docstrings() -> None:
    for cls in (DenseRetriever, BM25Retriever, HybridRetriever):
        assert cls.__doc__ is not None


def test_generator_and_reranker_docstrings() -> None:
    assert LlamaGenerator.__doc__ is not None
    assert "Llama" in LlamaGenerator.__doc__
    assert CrossEncoderReranker.__doc__ is not None


def test_embedder_dim_raises_before_load() -> None:
    cfg = RAGConfig()
    emb = Embedder(cfg)
    with pytest.raises(RuntimeError):
        _ = emb.dim


def test_prompt_template_enforces_citations_and_refusal() -> None:
    """Guardrail #5: the prompt must require chunk-ID citations and refusal."""
    assert "[chunk_id]" in RAG_PROMPT_TEMPLATE
    assert "Insufficient grounded evidence" in RAG_PROMPT_TEMPLATE
    assert "{query}" in RAG_PROMPT_TEMPLATE
    assert "{retrieved_chunks}" in RAG_PROMPT_TEMPLATE
    assert "{causal_context}" in RAG_PROMPT_TEMPLATE
