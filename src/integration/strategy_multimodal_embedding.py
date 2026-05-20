"""Multimodal-embedding integration strategy (ablation C of contribution C2).

Late-fuses disease and soil feature vectors (concat / weighted-sum /
cross-attention) and uses the fused embedding as the retrieval key
against a multi-vector ChromaDB collection.

Most aggressive of the three strategies; expected to win when textual
queries lose information.
"""

from __future__ import annotations

from src.integration.config import MultimodalEmbeddingStrategyConfig
from src.integration.context import MultimodalContext


class MultimodalEmbeddingStrategy:
    """Late-fusion embedding strategy.

    Parameters
    ----------
    config : MultimodalEmbeddingStrategyConfig
    """

    def __init__(self, config: MultimodalEmbeddingStrategyConfig) -> None:
        self.config = config

    def build_query_embedding(self, context: MultimodalContext) -> "object":
        """Fuse disease + soil features into a single retrieval vector.

        Returns
        -------
        np.ndarray
            Shape ``(projection_dim,)``.

        Raises
        ------
        NotImplementedError
            Phase 7 — Week 23.
        """
        raise NotImplementedError("Phase 7 — Week 23: implement multimodal embedding fusion.")
