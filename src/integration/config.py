"""Pydantic schema for the integration / joint context module.

Per master reference §13 (Integration) and contribution C2: the joint
disease + soil context module has three ablated integration strategies
(template, llm-mediated, multimodal-embedding). Per C5, the system does
**not** infer causal pathway from images — the user supplies it as
:class:`~src.integration.context.CausalContext`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.utils.config import BaseConfig

IntegrationStrategy = Literal["template", "llm_mediated", "multimodal_embedding"]


class TemplateStrategyConfig(BaseConfig):
    """Hyperparameters for the deterministic template integration strategy."""

    template_path: str = "configs/integration/templates/joint_advisory.j2"
    include_confidence: bool = True


class LLMMediatedStrategyConfig(BaseConfig):
    """Hyperparameters for the LLM-mediated joint reasoning strategy."""

    summary_max_tokens: int = Field(256, ge=32, le=2048)
    include_causal_context: bool = True


class MultimodalEmbeddingStrategyConfig(BaseConfig):
    """Hyperparameters for the late-fusion multimodal embedding strategy."""

    fusion_method: Literal["concat", "weighted_sum", "cross_attention"] = "concat"
    projection_dim: int = Field(512, ge=32)


class IntegrationConfig(BaseConfig):
    """Top-level integration config selecting one of three ablations."""

    strategy: IntegrationStrategy = "template"
    template: TemplateStrategyConfig = Field(default_factory=TemplateStrategyConfig)
    llm_mediated: LLMMediatedStrategyConfig = Field(default_factory=LLMMediatedStrategyConfig)
    multimodal_embedding: MultimodalEmbeddingStrategyConfig = Field(
        default_factory=MultimodalEmbeddingStrategyConfig
    )

    require_causal_context: bool = Field(
        default=True,
        description=(
            "If True, the integration step refuses to run unless a user-provided "
            "CausalContext is supplied. Enforces contribution C5."
        ),
    )
