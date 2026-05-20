"""LLM-mediated integration strategy (ablation B of contribution C2).

Uses the Llama generator to summarise the multimodal context into a
natural-language query before retrieval. Hypothesis: the LLM's prior
knowledge produces semantically richer queries than the template,
improving recall on classical-text phrasings.
"""

from __future__ import annotations

from src.integration.config import LLMMediatedStrategyConfig
from src.integration.context import MultimodalContext


class LLMMediatedStrategy:
    """LLM-summarised RAG query builder.

    Parameters
    ----------
    config : LLMMediatedStrategyConfig
    """

    def __init__(self, config: LLMMediatedStrategyConfig) -> None:
        self.config = config

    def build_query(self, context: MultimodalContext) -> str:
        """Use the LLM to compose a natural-language retrieval query.

        Raises
        ------
        NotImplementedError
            Phase 7 — Week 22.
        """
        raise NotImplementedError("Phase 7 — Week 22: implement LLMMediatedStrategy.build_query.")
