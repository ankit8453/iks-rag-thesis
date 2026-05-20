"""Template-based integration strategy (ablation A of contribution C2).

Renders a deterministic Jinja2 template using the disease + soil
predictions and the user's causal context. Used as the simple,
reproducible baseline against which the LLM-mediated and multimodal-
embedding strategies are compared.
"""

from __future__ import annotations

from src.integration.config import TemplateStrategyConfig
from src.integration.context import MultimodalContext


class TemplateStrategy:
    """Deterministic template renderer.

    Parameters
    ----------
    config : TemplateStrategyConfig
    """

    def __init__(self, config: TemplateStrategyConfig) -> None:
        self.config = config

    def build_query(self, context: MultimodalContext) -> str:
        """Render a RAG query string from the multimodal context.

        Raises
        ------
        NotImplementedError
            Phase 7 — Week 22 (joint integration).
        """
        raise NotImplementedError("Phase 7 — Week 22: implement TemplateStrategy.build_query.")
