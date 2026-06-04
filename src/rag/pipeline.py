"""Phase 7 RAG pipeline — wires the hybrid retriever and the grounded generator.

The pipeline is intentionally thin: it accepts a pre-built ChromaDB
collection plus an optional pre-constructed retriever and generator,
and exposes a single :meth:`RAGPipeline.answer` method that returns
``{answer, citations, retrieved}``.

The generator is **model-agnostic** via the ``generator`` constructor
arg: pass any class with a ``generate(query, retrieved_chunks) ->
GenerationResult``-shaped object. This is the swap-point for going
from Llama-3.1-8B to Llama-3.2-3B (or any other LLM), and the seam
where Phase 8 will inject vision-grounded context into the query.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from src.rag.generator import GroundedGenerator
from src.rag.retriever import HybridRetriever, RetrievedChunk
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

_LOGGER = get_logger(__name__)


@dataclass
class RAGAnswer:
    """Structured pipeline output.

    Attributes
    ----------
    answer
        The model's prose (possibly the §17 "insufficient evidence"
        refusal).
    citations
        Each ``"Source Text ch.X v.Y"`` string the answer cited.
    used_chunk_ids
        ``chunk_id`` of every retrieved chunk a citation matched.
    retrieved
        The full list of :class:`RetrievedChunk` returned by the
        retriever — useful for debugging and for the notebook's
        "show the chunks" cell.
    """

    answer: str
    citations: list[str]
    used_chunk_ids: list[str]
    retrieved: list[RetrievedChunk]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (chunks expanded) for JSON-friendly callers."""
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "used_chunk_ids": list(self.used_chunk_ids),
            "retrieved": [asdict(c) for c in self.retrieved],
        }


class RAGPipeline:
    """Top-level Phase 7 RAG pipeline.

    Parameters
    ----------
    collection
        ChromaDB collection populated by
        :func:`src.rag.corpus_loader.build_chroma`. Ignored if
        ``retriever`` is supplied.
    retriever
        Optional pre-built :class:`HybridRetriever`. If ``None`` the
        pipeline builds one from ``collection`` with the default
        toggles (dense + sparse + reranker all on).
    generator
        Anything with a ``generate(query, retrieved_chunks)`` method
        that returns an object with ``answer``, ``citations``, and
        ``used_chunk_ids`` attributes. Defaults to
        :class:`GroundedGenerator` (Llama-3.1-8B 4-bit).
    model_name
        Convenience arg forwarded to :class:`GroundedGenerator` when
        ``generator`` is ``None``. Pass
        ``"meta-llama/Llama-3.2-3B-Instruct"`` to swap LLM.
    default_k
        Default number of chunks returned by the retriever per
        :meth:`answer` call. Can be overridden per call.
    """

    def __init__(
        self,
        collection: "Collection | None" = None,
        *,
        retriever: HybridRetriever | None = None,
        generator: Any = None,
        model_name: str | None = None,
        default_k: int = 5,
    ) -> None:
        if retriever is None:
            if collection is None:
                raise ValueError(
                    "RAGPipeline needs either a collection or a pre-built retriever."
                )
            retriever = HybridRetriever(collection)
        self.retriever = retriever

        if generator is None:
            if model_name is not None:
                generator = GroundedGenerator(model_name=model_name)
            else:
                generator = GroundedGenerator()
        self.generator = generator

        self.default_k = int(default_k)
        _LOGGER.info(
            "RAGPipeline ready: retriever=%s generator=%s default_k=%d",
            type(self.retriever).__name__,
            type(self.generator).__name__,
            self.default_k,
        )

    def answer(self, query: str, k: int | None = None) -> RAGAnswer:
        """Retrieve top-``k`` chunks and ground a generation on them.

        Returns the same ``answer / citations / used_chunk_ids /
        retrieved`` shape regardless of which generator is wired in.
        """
        if k is None:
            k = self.default_k
        retrieved = self.retriever.retrieve(query, k=k)
        gen = self.generator.generate(query, retrieved)
        return RAGAnswer(
            answer=getattr(gen, "answer", ""),
            citations=list(getattr(gen, "citations", []) or []),
            used_chunk_ids=list(getattr(gen, "used_chunk_ids", []) or []),
            retrieved=retrieved,
        )


__all__ = [
    "RAGAnswer",
    "RAGPipeline",
]
