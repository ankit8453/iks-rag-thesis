"""Joint disease + soil context integration module.

Implements contribution C2 (joint context module with three ablated
integration strategies) and enforces contribution C5 (cause-conditional
retrieval; the system does NOT infer cause from images).
"""

from src.integration.causation import CausalContext, CausalPathway
from src.integration.causation_dataset import (
    MultiLabelImageDataset,
    make_olid_loaders,
)
from src.integration.config import IntegrationConfig
from src.integration.context import MultimodalContext
from src.integration.strategy_llm_mediated import LLMMediatedStrategy
from src.integration.strategy_multimodal_embedding import MultimodalEmbeddingStrategy
from src.integration.strategy_template import TemplateStrategy

__all__ = [
    "CausalContext",
    "CausalPathway",
    "IntegrationConfig",
    "LLMMediatedStrategy",
    "MultiLabelImageDataset",
    "MultimodalContext",
    "MultimodalEmbeddingStrategy",
    "TemplateStrategy",
    "make_olid_loaders",
]
