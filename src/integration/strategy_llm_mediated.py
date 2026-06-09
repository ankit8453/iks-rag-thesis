"""LLM-mediated integration strategy (Strategy B of contribution C2).

Re-uses the already-loaded Llama-3.1-8B (the Phase 7 ``GroundedGenerator``)
to *rewrite* the structured multimodal context into a single retrieval
query that BRIDGES modern vision labels (e.g. "rice blast",
"sandy_loam") to the descriptive / symptomatic vocabulary the IKS
corpus actually uses (e.g. "scorched leaves with whitish lesions",
"kunapajala", "soil-of-mixed-grain texture").

This bridge is the real reason Strategy B is the main contribution and
not just a fancier Strategy A: modern disease and soil labels are
absent from classical agricultural treatises, but the underlying
*phenomena* described in those treatises are very much present —
phrased in observational / vernacular terms. A blunt template query
("Organic treatment for Tomato Leaf Mold") underperforms one phrased
as a classical-text passage would have phrased it.

Hard rules baked into the prompt:

1. The LLM does NOT propose a treatment. It only reformulates the query.
2. It uses no information beyond the structured context provided.
3. It emits exactly one retrieval query, one sentence, no commentary.
4. If a causal pathway is supplied, the rewrite respects it; if
   ``UNKNOWN``, no pathway clause is added.
"""

from __future__ import annotations

from typing import Any

from src.integration.causation import CausalPathway
from src.integration.config import LLMMediatedStrategyConfig
from src.integration.context import MultimodalContext

# Default low-temperature, fixed-seed config — Strategy B must be
# deterministic for the qualitative comparison to be reproducible.
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_SEED: int = 42

_PATHWAY_HINTS: dict[CausalPathway, str] = {
    CausalPathway.SOIL_DRIVEN: "the farmer suspects the cause is rooted in the soil",
    CausalPathway.PEST_VECTOR: "the farmer suspects insects or pests are responsible",
    CausalPathway.CONTAGION: "the farmer suspects the affliction is spreading from neighbouring plants",
    CausalPathway.UNKNOWN: "",
}

_SYSTEM_INSTRUCTIONS: str = (
    "You are reformulating a single agricultural query so it can be matched "
    "against an Indian Knowledge Systems (IKS) corpus of classical Sanskrit "
    "treatises (Vrikshayurveda, Brihat Samhita, Krishi Parashara, "
    "Upavanavinoda) in English translation.\n"
    "\n"
    "RULES:\n"
    "1. Output exactly ONE retrieval query, one sentence, no commentary.\n"
    "2. Do NOT propose a treatment, remedy, or protocol. Only describe what "
    "we are looking for.\n"
    "3. Bridge modern vision labels (disease names like 'leaf mold', soil "
    "labels like 'Alluvial') to descriptive, symptom-oriented, or classical "
    "vocabulary that the IKS corpus would actually use (e.g. 'scorched "
    "leaves with whitish spots', 'fertile riverine soil', 'pot-irrigation', "
    "'kunapajala-style treatment').\n"
    "4. Stay strictly within the structured context provided. Invent no "
    "extra symptoms, no extra soil properties, and no farmer narrative "
    "beyond what is supplied.\n"
    "5. If a suspected cause is supplied, work it into the query (e.g. with "
    "emphasis on soil health). If the cause is 'unknown', do not mention "
    "cause at all.\n"
)


class LLMMediatedStrategy:
    """Llama-mediated retrieval-query builder (Strategy B).

    Parameters
    ----------
    config
        :class:`LLMMediatedStrategyConfig`.
    """

    def __init__(self, config: LLMMediatedStrategyConfig) -> None:
        self.config = config

    # ------------------------------------------------------------- #

    def build_query(self, context: MultimodalContext, llm: Any) -> str:
        """Use the passed-in Llama generator to rewrite ``context`` as a query.

        Parameters
        ----------
        context
            The :class:`MultimodalContext` produced upstream.
        llm
            An already-loaded :class:`~src.rag.generator.GroundedGenerator`
            (the same Llama instance Phase 7 builds at notebook setup).
            We reuse it to avoid a second 5-min model load.

        Returns
        -------
        str
            One reformulated retrieval query, ready to feed into the
            Phase 7 ``HybridRetriever`` (via ``RAGPipeline.retrieve``).
        """
        prompt = self._build_prompt(context)
        text = self._invoke_llm(llm, prompt).strip()
        return _first_sentence(text)

    # ------------------------------------------------------------- #

    def _build_prompt(self, context: MultimodalContext) -> str:
        """Compose the deterministic Strategy-B prompt from ``context``.

        Public-ish helper so :file:`tests/integration/test_compare_smoke.py`
        can introspect prompt construction without instantiating Llama.
        """
        disease = context.disease_pred.class_name
        disease_conf = float(context.disease_pred.confidence)
        soil_type = context.soil_pred.soil_type
        moisture = context.soil_pred.moisture_appearance
        texture = context.soil_pred.texture
        crop = context.crop_type.strip() or "an unspecified crop"
        pathway = context.causal_context.pathway
        pathway_hint = _PATHWAY_HINTS.get(pathway, "")

        causal_line = (
            f"- Suspected cause: {pathway_hint}.\n"
            if pathway_hint
            else "- Suspected cause: not specified.\n"
        )
        return (
            f"{_SYSTEM_INSTRUCTIONS}\n"
            "STRUCTURED CONTEXT:\n"
            f"- Crop: {crop}\n"
            f"- Vision-detected disease (modern label): {disease} "
            f"(confidence {disease_conf:.2f})\n"
            f"- Soil type (modern label): {soil_type}\n"
            f"- Soil moisture appearance: {moisture}\n"
            f"- Soil texture: {texture}\n"
            f"{causal_line}\n"
            "ONE RETRIEVAL QUERY (one sentence, no commentary):"
        )

    # ------------------------------------------------------------- #

    def _invoke_llm(self, llm: Any, prompt: str) -> str:
        """Best-effort generator-agnostic LLM call.

        Tries (in order):

        - ``llm.complete(prompt)`` — preferred. Phase 7's
          :class:`~src.rag.generator.GroundedGenerator` exposes this
          method specifically for Phase 8 Strategy B; it runs the
          tokenizer + model on a *plain* prompt with NO §17 grounding
          wrapper (which would otherwise inject system instructions
          for grounded advisory generation and poison the rewrite).
        - ``llm.generate(prompt)`` — fallback for plain HF / OpenAI
          style generators that take a string and return a string.
          Skipped automatically if the call raises ``TypeError`` (e.g.
          GroundedGenerator's ``generate(query, retrieved_chunks)``
          signature mismatch).
        - ``llm.__call__(prompt)`` — last resort.

        Each method is expected to return either a ``str`` or an object
        with a ``.text`` attribute.
        """
        for attr in ("complete", "generate", "__call__"):
            fn = getattr(llm, attr, None)
            if fn is None:
                continue
            try:
                out = fn(prompt)
            except TypeError:
                # Wrong signature (e.g. GroundedGenerator.generate
                # wants (query, retrieved_chunks)). Skip and try the
                # next interface.
                continue
            if isinstance(out, str):
                return out
            text = getattr(out, "text", None)
            if isinstance(text, str):
                return text
            return str(out)
        raise TypeError(
            "LLM object has no usable .complete / .generate / __call__ "
            "method; cannot run Strategy B. Pass a generator that exposes "
            ".complete(prompt: str) -> str."
        )


# --------------------------------------------------------------------- #


def _first_sentence(text: str) -> str:
    """Strip the LLM output to a single sentence so the retriever gets
    a clean query string and not a paragraph."""
    # Drop chat-template artefacts.
    for marker in ("ONE RETRIEVAL QUERY:", "Query:", "QUERY:"):
        if marker in text:
            text = text.split(marker, 1)[1]
    cleaned = text.strip().strip('"').strip()
    if not cleaned:
        return ""
    # First terminator wins.
    for sep in (". ", ".\n", "?\n", "?\t", "\n\n"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0]
    cleaned = cleaned.strip().rstrip(".").strip()
    return cleaned + "." if cleaned else ""
