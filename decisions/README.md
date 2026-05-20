# Architecture Decision Records (ADRs)

Lightweight ADR format borrowed from Michael Nygard. One file per
decision. Each ADR has:

- **Status** — Proposed / Accepted / Superseded by [link]
- **Context** — what's the situation that prompted this decision?
- **Decision** — what did we choose?
- **Consequences** — what follows, including the downsides?

Decisions are append-only. To change one, write a new ADR that
supersedes it.

## Index

- [0001 — EfficientNet backbones over ResNet50](0001-efficientnet-backbones.md)
- [0002 — Pydantic v2 over Hydra for configs](0002-pydantic-over-hydra.md)
- [0003 — No LangChain / LlamaIndex in the RAG layer](0003-no-langchain-in-rag.md)
