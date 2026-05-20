# ADR-0003 — No LangChain / LlamaIndex in the RAG layer

## Status

Accepted — 2026-05-20. Supersedes the Week 1 `requirements.txt` that
included `langchain` and `langchain-community`.

## Context

The RAG layer is the most-evaluated part of the thesis. Reviewers will
want to know exactly what the retriever, reranker, and generator do.

LangChain and LlamaIndex both provide convenient RAG abstractions
(retrievers, chains, vectorstores), but they:

1. Change their public API frequently — a `0.0.x → 0.1.x` bump has
   broken downstream code multiple times in the project history.
2. Hide important behaviour behind chain configuration (which tokenizer
   was used, which template was rendered, exactly when context was
   truncated) — exactly the things a reviewer needs to inspect.
3. Make reproducibility harder for somebody trying to re-run the thesis
   six months later.

The pieces we actually need (sentence-transformers, BGE reranker,
`rank_bm25`, ChromaDB, transformers + bitsandbytes) are all small,
direct, and well-documented. Wrapping them ourselves in
`src/rag/{embedder,retriever,reranker,generator}.py` is more code, but
the code is the spec.

## Decision

The RAG layer is implemented as plain Python wrappers around:

- `sentence_transformers` for embeddings and the cross-encoder reranker.
- `rank_bm25` for sparse retrieval.
- `chromadb` for the persistent vector store.
- `transformers` + `bitsandbytes` for the Llama-3.1-8B generator.

LangChain and LlamaIndex are **not** dependencies. They may be used
later for sketch / prototype scripts kept out of the package, but never
in `src/`.

## Consequences

- More code in `src/rag/`. Mitigated by the small surface (one class
  per role) and shared `RAGConfig`.
- We have to track upstream changes in `transformers`, `sentence-
  transformers`, and `chromadb` ourselves. These three have stable
  enough APIs that this is manageable.
- The prompt template lives in version control as a Python constant
  (`src/rag/prompts.py::RAG_PROMPT_TEMPLATE`) rather than a YAML file
  hidden inside a chain — reviewable in `git diff`.
- The eval harness (`src/eval/citation_verification.py`,
  `src/eval/ragas_eval.py`) does not need to thread through LangChain
  callbacks — it just consumes our own dataclasses.

## References

- Locked Stack table in `WEEK2_PROMPT.md`.
