"""Llama-3.1-8B generator wrapper.

Wraps the HuggingFace ``transformers`` pipeline plus
``bitsandbytes`` 4-bit quantisation so the rest of the codebase only sees
``LlamaGenerator.generate(prompt) -> str``. Per supervisor guardrail #5,
the prompt template (in :mod:`src.rag.prompts`) instructs the model to
cite retrieved chunk IDs and the eval harness verifies those citations
actually appear in the retrieved context.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.config import RAGConfig
from src.rag.retriever import RetrievedChunk


@dataclass
class GenerationResult:
    """LLM output plus the citations it claims to support.

    Attributes
    ----------
    answer : str
        The generated answer text.
    cited_chunk_ids : list[str]
        Chunk IDs the model cited inside ``answer``. Verified downstream
        by :func:`src.eval.citation_verification.verify_citations_in_context`.
    prompt_tokens : int
        Token count of the rendered prompt (for cost / context budgeting).
    completion_tokens : int
        Token count of the generated answer.
    """

    answer: str
    cited_chunk_ids: list[str]
    prompt_tokens: int
    completion_tokens: int


class LlamaGenerator:
    """Llama-3.1-8B-Instruct generator with 4-bit quantisation.

    Parameters
    ----------
    config : RAGConfig
        Provides ``llm_model``, ``llm_quantization``,
        ``llm_max_new_tokens``, ``llm_temperature``.

    Notes
    -----
    Phase 4 implementation (Week 14) will:

    1. Build a ``BitsAndBytesConfig`` (``load_in_4bit=True``,
       ``bnb_4bit_compute_dtype=torch.float16``).
    2. Load the tokenizer and model from ``config.llm_model``.
    3. Render the :data:`~src.rag.prompts.RAG_PROMPT_TEMPLATE` with the
       query, retrieved chunks, and any
       :class:`~src.integration.context.CausalContext` provided by the
       user.
    4. Extract cited chunk IDs from the generated answer with a regex.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._model: object | None = None
        self._tokenizer: object | None = None

    def load(self) -> None:
        """Load tokenizer and (quantised) model weights.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: load Llama-3.1-8B 4-bit.")

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        causal_context: "object | None" = None,
    ) -> GenerationResult:
        """Generate a cited answer.

        Parameters
        ----------
        query : str
            The user's question (possibly augmented by disease/soil pred).
        chunks : list[RetrievedChunk]
            Reranked top-k context.
        causal_context : CausalContext | None
            User-provided causal pathway. Per contribution C5 this is
            never inferred from images.

        Returns
        -------
        GenerationResult

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 14.
        """
        raise NotImplementedError("Phase 4 — Week 14: implement LlamaGenerator.generate.")
