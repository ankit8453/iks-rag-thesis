"""RAGAS evaluation wrapper.

Wraps :mod:`ragas` so the evaluation harness sees a single ``run_ragas_
evaluation`` function that takes a list of ``(query, answer, contexts,
ground_truth)`` records and returns a per-metric mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.eval.config import EvalConfig


@dataclass
class RAGEvalSample:
    """One evaluation row.

    Attributes
    ----------
    query : str
    answer : str
        The system-generated answer.
    contexts : list[str]
        Retrieved chunks shown to the generator.
    ground_truth : str | None
        Expert-written reference answer, when available. Required for
        ``answer_correctness`` but not for ``faithfulness``.
    """

    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@dataclass
class RAGASScores:
    """Mean RAGAS scores across the evaluation set."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    context_relevancy: float | None = None
    per_sample: list[dict[str, float]] = field(default_factory=list)


def run_ragas_evaluation(
    samples: list[RAGEvalSample],
    config: EvalConfig,
    output_path: Path | None = None,
) -> RAGASScores:
    """Compute RAGAS scores over ``samples``.

    Parameters
    ----------
    samples : list[RAGEvalSample]
        Evaluation set. May come from
        :func:`src.eval.expert_annotation.load_annotations`.
    config : EvalConfig
        Provides ``config.ragas.metrics`` (subset to compute).
    output_path : Path, optional
        Write the per-sample table here as CSV.

    Returns
    -------
    RAGASScores

    Raises
    ------
    NotImplementedError
        Phase 9 — Week 30 (RAG evaluation).
    """
    raise NotImplementedError("Phase 9 — Week 30: implement RAGAS evaluation wrapper.")
