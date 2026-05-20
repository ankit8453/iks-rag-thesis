"""Pydantic schema for the evaluation harness.

Per master reference §15 (Evaluation) and contribution C3: RAG faithfulness
is measured with RAGAS plus domain-expert annotations. Per C4, hallucination
on IKS sources is quantified; per supervisor guardrail #3, every CV metric
report must include per-class precision/recall/F1 and a confusion matrix.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.utils.config import BaseConfig

RAGASMetric = Literal[
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "context_relevancy",
]

_DEFAULT_RAGAS_METRICS: list[RAGASMetric] = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


class CVMetricsConfig(BaseConfig):
    """Configuration for the classification metrics reporter."""

    report_per_class: bool = True
    save_confusion_matrix: bool = True
    average: Literal["macro", "weighted", "micro"] = "macro"


class RAGASConfig(BaseConfig):
    """Subset of RAGAS metrics to compute on each evaluation run."""

    metrics: list[RAGASMetric] = Field(
        default_factory=lambda: list(_DEFAULT_RAGAS_METRICS),
    )
    sample_size: int | None = Field(
        default=None,
        description="If set, evaluate on a random subsample of this size (for cost).",
    )


class ExpertAnnotationConfig(BaseConfig):
    """Where to read / write expert annotations.

    Paths are relative to ``PROJECT_ROOT``.
    """

    annotation_csv: str = "results/expert_annotations.csv"
    annotators: list[str] = Field(default_factory=list)


class EvalConfig(BaseConfig):
    """Top-level evaluation configuration."""

    cv: CVMetricsConfig = Field(default_factory=CVMetricsConfig)
    ragas: RAGASConfig = Field(default_factory=RAGASConfig)
    expert: ExpertAnnotationConfig = Field(default_factory=ExpertAnnotationConfig)
    seed: int = Field(42, ge=0)
