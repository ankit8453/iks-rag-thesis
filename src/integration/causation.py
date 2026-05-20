"""Causal context types.

The system DOES NOT infer causal pathway from images. The
:class:`CausalPathway` enum and :class:`CausalContext` dataclass exist so
the user can supply their best guess (soil-driven, pest vector, contagion,
unknown). This is the foundation of contribution C5 (cause-conditional
retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CausalPathway(str, Enum):
    """Pathways the user can attribute a plant problem to.

    These categories were chosen with the supervisor to give the RAG
    retriever a meaningful filter without forcing the system to guess
    causation from imagery.
    """

    SOIL_DRIVEN = "soil_driven"
    PEST_VECTOR = "pest_vector"
    CONTAGION = "contagion"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CausalContext:
    """User-supplied causal hypothesis.

    Attributes
    ----------
    pathway : CausalPathway
        The high-level pathway the farmer suspects.
    notes : str | None
        Free-text narrative; optional. Useful for downstream query
        expansion (Phase 4).
    """

    pathway: CausalPathway
    notes: str | None = None
