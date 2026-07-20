"""Phase 7 grounded-advisor generator — Llama-3.1-8B 4-bit + master plan §17 prompt.

The class :class:`GroundedGenerator` loads ``meta-llama/Llama-3.1-8B-
Instruct`` via ``transformers`` + ``bitsandbytes`` 4-bit (nf4,
double-quant, bf16 compute) and exposes a single
:meth:`GroundedGenerator.generate` method that takes a query and the
retrieved chunks and returns ``{answer, citations, used_chunk_ids}``.

System prompt (master plan §17, locked):

- The advisor is grounded strictly in the retrieved classical-text
  passages.
- It must answer ONLY from those passages; if they lack the answer it
  must say so plainly.
- Every factual claim must cite the source text plus the chapter and
  verse / section.
- Recommendations are step-by-step organic protocols; no modern
  agro-chemical interventions get introduced unless a retrieved passage
  explicitly names them.

Citations use the ``source_text + chapter + verse/section`` triple
(NOT the chunk_id form from the deprecated Phase-4 prompts.py), so the
LLM cites human-legible source coordinates. The chunk_id round-trip is
preserved in the structured result for downstream traceability.

``LlamaGenerator`` is kept as an alias of :class:`GroundedGenerator` so
any leftover ``from src.rag.generator import LlamaGenerator`` imports
keep working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.rag.retriever import RetrievedChunk
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

_LOGGER = get_logger(__name__)


DEFAULT_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_SEED: int = 42

# Master plan §17 — grounded-advisor system prompt. The rule numbering
# mirrors the four bullets in the §17 spec; the citation format
# explicitly names source + chapter + verse so the model emits
# human-legible coordinates.
SYSTEM_PROMPT_V17: str = (
    "You are an Indian-Knowledge-Systems agricultural advisor grounded "
    "strictly in classical Sanskrit texts (Vrikshayurveda, Brihat "
    "Samhita, and related works) supplied to you as retrieved passages. "
    "You MUST follow every rule below.\n"
    "\n"
    "1. ANSWER ONLY FROM THE RETRIEVED PASSAGES. Do not draw on outside "
    "knowledge. If the retrieved passages do not contain enough "
    "information to answer the user's question, respond exactly: "
    "\"The retrieved classical-text passages do not contain enough "
    "information to answer this question. Please consult a qualified "
    "agricultural expert.\"\n"
    "1a. SCOPE OF RULE 1 — THESE TEXTS ARE GENERAL AND SYMPTOM-BASED, NOT "
    "CROP-SPECIFIC. They prescribe by observed symptom and its underlying "
    "cause, and such a remedy applies across plants showing that symptom. "
    "Therefore a passage that addresses the OBSERVED CONDITION does count as "
    "sufficient evidence even when it never names the user's crop. Do NOT "
    "refuse merely because the crop is absent from the passages. Refuse only "
    "when no retrieved passage addresses the condition at all. When you rely "
    "on a general passage, say so plainly (e.g. \"the texts prescribe this for "
    "this symptom generally\") rather than implying it was written for that "
    "crop.\n"
    "2. CITE EVERY CLAIM. After each factual statement, cite the source "
    "text, chapter, and verse or section in the form "
    "[Source Text, ch.<chapter>, v.<verse_or_section>]. Example: "
    "[Vrikshayurveda, ch.full, v.160.3]. Cite only sources that "
    "actually appear in the retrieved passages.\n"
    "3. STRUCTURE AS A STEP-BY-STEP ORGANIC PROTOCOL when the user asks "
    "for treatment / remedy / procedure. Use numbered steps. Each step "
    "must carry at least one citation.\n"
    "4. DO NOT INTRODUCE TREATMENTS, INGREDIENTS, OR MODERN AGRO-"
    "CHEMICALS that are absent from the retrieved passages. If a "
    "modern intervention is needed, defer it to expert consultation "
    "rather than inventing one.\n"
)


# --------------------------------------------------------------------- #
# Result + helpers
# --------------------------------------------------------------------- #


@dataclass
class GenerationResult:
    """Structured output of :meth:`GroundedGenerator.generate`."""

    answer: str
    citations: list[str]
    used_chunk_ids: list[str]
    raw_completion: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


# Regex matches the cited form [Source Text, ch.<chapter>, v.<verse_or_section>]
# with tolerant whitespace.
_CITATION_RE = re.compile(
    r"\[\s*(?P<src>[^,\]]+?)\s*,\s*ch\.\s*(?P<chap>[^,\]]+?)\s*"
    r",\s*v\.\s*(?P<verse>[^\]]+?)\s*\]",
    re.IGNORECASE,
)


def _format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a labelled context block for the LLM."""
    lines: list[str] = []
    for i, ch in enumerate(chunks, 1):
        meta = ch.metadata or {}
        src = meta.get("source_text", "?")
        chap = meta.get("chapter", "?")
        verse = meta.get("verse_or_section", "?")
        translator = meta.get("translator", "")
        tag = f"[Source {i}] {src}, ch.{chap}, v.{verse}"
        if translator:
            tag += f" (tr. {translator})"
        body = (ch.text or "").strip()
        lines.append(f"{tag}\n{body}")
    return "\n\n".join(lines)


def extract_citations(answer: str) -> list[str]:
    """Pull ``[Source, ch.X, v.Y]`` citations out of a generated answer.

    Returns a deduplicated list in first-seen order, each formatted as
    ``"Source Text ch.X v.Y"`` so callers can join with retrieved
    chunk metadata.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _CITATION_RE.finditer(answer or ""):
        key = (
            f"{m.group('src').strip()} ch.{m.group('chap').strip()} "
            f"v.{m.group('verse').strip()}"
        )
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _match_citations_to_chunks(
    citations: list[str],
    chunks: list[RetrievedChunk],
) -> list[str]:
    """Map each citation back to the chunk_id whose metadata it matches."""
    used: list[str] = []
    for cite in citations:
        for ch in chunks:
            meta = ch.metadata or {}
            label = (
                f"{meta.get('source_text', '?')} ch.{meta.get('chapter', '?')} "
                f"v.{meta.get('verse_or_section', '?')}"
            )
            if cite.lower() == label.lower():
                if ch.chunk_id not in used:
                    used.append(ch.chunk_id)
                break
    return used


# --------------------------------------------------------------------- #
# GroundedGenerator
# --------------------------------------------------------------------- #


class GroundedGenerator:
    """Llama-3.1-8B 4-bit grounded advisor.

    Parameters
    ----------
    model_name
        HF Hub model id. Default ``meta-llama/Llama-3.1-8B-Instruct``
        (gated — the operator must have accepted the licence on HF).
        Swappable to ``meta-llama/Llama-3.2-3B-Instruct`` for tighter
        VRAM budgets via the same constructor.
    max_new_tokens, temperature, seed
        Decoding controls. Defaults match master plan §17 and the
        Phase 7 prompt's locked decisions (T=0.2, deterministic seed).
    load_in_4bit
        If True (default), loads under bitsandbytes nf4 4-bit. If
        False, loads in bf16 (24+ GB VRAM only).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.load_in_4bit = bool(load_in_4bit)

        self._tokenizer = None
        self._model = None
        # Lazy load — keeps module-import time low and lets tests stub
        # the generate path without paying the model-download cost.

    # ----- lazy load ------------------------------------------------- #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        _LOGGER.info("Loading tokenizer for %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            _LOGGER.info("Loading %s in 4-bit (nf4 + double-quant) ...", self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            _LOGGER.info("Loading %s in bf16 (no 4-bit quant) ...", self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self._model.eval()

    # ----- prompt builder ------------------------------------------- #

    def build_prompt(
        self, query: str, retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """Render the chat-template prompt fed to the model.

        Public so tests + the notebook can inspect what gets sent.
        """
        context_block = _format_context_block(retrieved_chunks)
        user_block = (
            f"USER QUESTION:\n{query}\n\n"
            f"RETRIEVED PASSAGES:\n{context_block}\n\n"
            "Following the four rules in the system prompt, answer the user's "
            "question. Cite every factual claim with the [Source Text, ch.X, "
            "v.Y] format."
        )
        if self._tokenizer is None:
            # Best-effort plain rendering for tests that monkeypatch
            # generate without loading the real tokenizer.
            return f"<<SYS>>\n{SYSTEM_PROMPT_V17}\n<</SYS>>\n\n{user_block}"
        try:
            return self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT_V17},
                    {"role": "user", "content": user_block},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            return f"<<SYS>>\n{SYSTEM_PROMPT_V17}\n<</SYS>>\n\n{user_block}"

    # ----- public API ------------------------------------------------ #

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        """Generate a grounded answer from ``retrieved_chunks``.

        Returns
        -------
        GenerationResult
            ``answer`` (the model's prose), ``citations`` (each
            ``"Source ch.X v.Y"`` string the answer cited), and
            ``used_chunk_ids`` (the subset of ``retrieved_chunks``
            whose ``(source, chapter, verse)`` matched a citation).
        """
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        prompt = self.build_prompt(query, retrieved_chunks)

        # Deterministic seeding so the demo cells are reproducible.
        torch.manual_seed(self.seed)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                pad_token_id=(self._tokenizer.eos_token_id or 0),
            )
        # Strip the prompt tokens.
        completion_tokens = out[0][inputs["input_ids"].shape[1]:]
        completion = self._tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return self.postprocess(completion, retrieved_chunks)

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run the loaded LM on a plain prompt and return decoded text.

        Phase 8 Strategy B (LLM-mediated query rewrite) needs a raw
        prompt → text path that bypasses the §17 grounding wrapper —
        it is *reformulating a retrieval query*, not answering a
        grounded question, so the system prompt + retrieved-passages
        block from :meth:`generate` would actively poison the output.

        Determinism mirrors :meth:`generate`: same seed, same
        temperature, same sampling toggle (off when ``temperature == 0``).
        """
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        torch.manual_seed(self.seed)
        max_new = int(max_new_tokens or min(256, self.max_new_tokens))

        # Use the tokenizer's chat-template if available so the model
        # sees the prompt in its native instruction-tuned format. Falls
        # back to a plain raw string for tests that monkeypatch the
        # tokenizer.
        rendered: str
        if self._tokenizer is None:
            rendered = prompt
        else:
            try:
                rendered = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                rendered = prompt

        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                pad_token_id=(self._tokenizer.eos_token_id or 0),
            )
        completion_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(completion_tokens, skip_special_tokens=True)

    def postprocess(
        self,
        raw_completion: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        """Pull citations out of ``raw_completion`` and link them to chunks.

        Kept as a method so tests can exercise the parsing without
        loading the model.
        """
        citations = extract_citations(raw_completion)
        used_chunk_ids = _match_citations_to_chunks(citations, retrieved_chunks)
        return GenerationResult(
            answer=raw_completion.strip(),
            citations=citations,
            used_chunk_ids=used_chunk_ids,
            raw_completion=raw_completion,
        )


# Backwards-compat alias for the pre-Phase-7 import path used by the
# deprecated stub in src.rag.__init__.
LlamaGenerator = GroundedGenerator


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_SEED",
    "DEFAULT_TEMPERATURE",
    "GenerationResult",
    "GroundedGenerator",
    "LlamaGenerator",
    "SYSTEM_PROMPT_V17",
    "extract_citations",
]
