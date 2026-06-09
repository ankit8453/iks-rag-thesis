"""Joint disease + soil context integration module (Phase 8).

Implements contribution C2 (joint context module with three ablated
integration strategies — template, LLM-mediated, multimodal-embedding)
and enforces contribution C5 (cause-conditional retrieval; the system
does NOT infer cause from images — pathway is user-supplied).

Public API:

- :class:`MultimodalContext` — frozen dataclass bundling disease + soil
  predictions, optional penultimate embeddings, crop, and user-supplied
  :class:`CausalContext`.
- :func:`build_multimodal_context` — orchestrates Phase 5 disease + Phase 6
  soil inference into a populated ``MultimodalContext``.
- :class:`TemplateStrategy` — deterministic Jinja2-free template
  (strategy A; transparent baseline).
- :class:`LLMMediatedStrategy` — Llama-mediated rewrite that bridges
  modern vision labels to classical-text vocabulary (strategy B; the
  main contribution).
- :class:`MultimodalEmbeddingStrategy` + :class:`MultimodalProjector` —
  weak-signal projection of fused visual embeddings into the corpus
  embedding space (strategy C; honest ablation showing the modality gap).
- :func:`run_all_strategies`, :func:`qualitative_compare` — side-by-side
  driver + reader.
"""

from src.integration.causation import CausalContext, CausalPathway
from src.integration.compare import qualitative_compare, run_all_strategies
from src.integration.config import IntegrationConfig
from src.integration.context import MultimodalContext, build_multimodal_context
from src.integration.strategy_llm_mediated import LLMMediatedStrategy
from src.integration.strategy_multimodal_embedding import (
    MultimodalEmbeddingStrategy,
    MultimodalProjector,
)
from src.integration.strategy_template import TemplateStrategy

__all__ = [
    "CausalContext",
    "CausalPathway",
    "IntegrationConfig",
    "LLMMediatedStrategy",
    "MultimodalContext",
    "MultimodalEmbeddingStrategy",
    "MultimodalProjector",
    "TemplateStrategy",
    "build_multimodal_context",
    "qualitative_compare",
    "run_all_strategies",
]
