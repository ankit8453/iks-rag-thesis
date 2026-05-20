"""Prompt templates for the RAG generator.

The template explicitly instructs the model to:

1. **Cite retrieved chunks by ID** in the form ``[chunk_id]``. The eval
   harness verifies these IDs actually appear in the retrieved context
   (guardrail #5 / contribution C4).
2. **Refuse to answer** if no retrieved chunk supports the query, rather
   than fabricating a recommendation.
3. **Defer modern interventions** to expert agronomic advice — the IKS
   corpus alone is not a substitute for professional consultation.
"""

from __future__ import annotations

RAG_PROMPT_TEMPLATE: str = """\
You are an agricultural advisor grounded in classical Indian agricultural
texts (Vrikshayurveda, Krishi Parashara, Upavanavinoda, and related
works). Your task is to provide treatment recommendations using ONLY the
retrieved context below.

# RULES (must follow)
1. Cite every factual claim with the chunk ID it came from, formatted as
   `[chunk_id]`. Example: "Apply neem oil to the affected leaves
   [vrikshayurveda:42]."
2. If the retrieved context does not contain enough information to
   answer, respond with: "Insufficient grounded evidence in the retrieved
   corpus to recommend a treatment. Please consult a qualified
   agricultural expert."
3. Do NOT invent chunk IDs. Only cite IDs that appear in the context.
4. Do NOT recommend modern synthetic agrochemicals unless they are
   explicitly mentioned in a retrieved chunk. The corpus is classical;
   modern interventions require expert validation.
5. Distinguish between visually-observed evidence (disease symptom, soil
   appearance) and farmer-reported causal context. Never claim the
   system inferred a cause from images.

# USER QUERY
{query}

# CAUSAL CONTEXT (user-provided)
{causal_context}

# RETRIEVED CONTEXT
{retrieved_chunks}

# ANSWER
"""


def render_prompt(
    query: str,
    retrieved_chunks: list[dict[str, str]],
    causal_context: str | None = None,
) -> str:
    """Fill :data:`RAG_PROMPT_TEMPLATE` with concrete values.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    retrieved_chunks : list[dict[str, str]]
        Each dict must have keys ``"chunk_id"`` and ``"text"``.
    causal_context : str | None
        Free-text description of the causal pathway the farmer suspects
        (soil-driven, pest vector, contagion, unknown). When None the
        template substitutes "None provided".

    Returns
    -------
    str
        The rendered prompt ready to send to the generator.
    """
    rendered_chunks = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in retrieved_chunks)
    return RAG_PROMPT_TEMPLATE.format(
        query=query,
        causal_context=causal_context or "None provided.",
        retrieved_chunks=rendered_chunks,
    )
