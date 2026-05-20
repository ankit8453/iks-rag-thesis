"""Retrieval-Augmented Generation pipeline over IKS corpora.

Implements contributions C1 (corpus), C3 (faithfulness-aware RAG), C4
(hallucination measurement) and C5 (cause-conditional retrieval). No
LangChain or LlamaIndex (ADR-0003); the orchestration is plain Python
wrapping HuggingFace + chromadb + rank-bm25.

Public surface:

- :class:`IKSCorpus` — chunked, metadata-tagged corpus loader
- :class:`Chunker` — paragraph + sliding-window chunkers
- :class:`DenseRetriever`, :class:`BM25Retriever`, :class:`HybridRetriever`
- :class:`CrossEncoderReranker`
- :class:`LlamaGenerator` — Llama-3.1-8B 4-bit
- :data:`RAG_PROMPT_TEMPLATE` — citation-enforcing prompt
"""

from src.rag.chunker import Chunk, Chunker
from src.rag.config import ChunkerConfig, RAGConfig
from src.rag.corpus_loader import IKSCorpus
from src.rag.embedder import Embedder
from src.rag.generator import LlamaGenerator
from src.rag.hybrid import HybridRetriever
from src.rag.prompts import RAG_PROMPT_TEMPLATE
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import BM25Retriever, DenseRetriever, RetrievedChunk

__all__ = [
    "BM25Retriever",
    "Chunk",
    "Chunker",
    "ChunkerConfig",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Embedder",
    "HybridRetriever",
    "IKSCorpus",
    "LlamaGenerator",
    "RAGConfig",
    "RAG_PROMPT_TEMPLATE",
    "RetrievedChunk",
]
