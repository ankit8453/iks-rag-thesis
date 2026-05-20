"""Evaluation harness for CV and RAG components.

Implements contributions C3 (faithfulness-aware RAG evaluation) and C4
(hallucination measurement on IKS sources). All CV metric utilities here
report per-class precision/recall/F1 plus a confusion matrix — never
just headline accuracy (supervisor guardrail #3).
"""

from src.eval.citation_verification import (
    CitationReport,
    verify_citations_in_context,
)
from src.eval.config import EvalConfig
from src.eval.cv_metrics import ClassificationReport, compute_per_class_metrics
from src.eval.expert_annotation import ExpertAnnotation, load_annotations
from src.eval.ragas_eval import run_ragas_evaluation

__all__ = [
    "CitationReport",
    "ClassificationReport",
    "EvalConfig",
    "ExpertAnnotation",
    "compute_per_class_metrics",
    "load_annotations",
    "run_ragas_evaluation",
    "verify_citations_in_context",
]
