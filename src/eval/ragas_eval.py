"""RAGAS evaluation wrapper.

Wraps :mod:`ragas` so the harness sees one ``run_ragas_evaluation`` function
taking ``(query, answer, contexts, ground_truth)`` records and returning
per-metric means.

**Cost note (why this is wired to local models).** RAGAS defaults to an OpenAI
judge, which would bill per evaluation run. This project runs on a tight budget,
so :func:`run_ragas_evaluation` accepts a ``judge_llm`` / ``judge_embeddings``
pair — pass the Llama already loaded on Colab and the evaluation is free and
fully reproducible offline. Nothing here ever calls a paid API implicitly.

**Which metrics need what.** ``faithfulness`` and ``answer_relevancy`` need only
the question, answer and retrieved contexts, so they can be computed now.
``context_precision`` / ``context_recall`` / ``answer_correctness`` compare
against a reference answer, so they are skipped unless ``ground_truth`` is
present — and skipped metrics are reported as ``None`` rather than 0.0, because
"not measured" is not the same as "scored zero".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.config import EvalConfig
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

#: Metrics that require a reference ("ground truth") answer to be meaningful.
NEEDS_GROUND_TRUTH: frozenset[str] = frozenset({
    "context_precision", "context_recall", "answer_correctness",
})


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
    #: Metrics that were requested but could not be computed, and why.
    skipped: dict[str, str] = field(default_factory=dict)

    def as_row(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
        }


def select_metrics(
    requested: list[str], samples: list[RAGEvalSample],
) -> tuple[list[str], dict[str, str]]:
    """Split requested metrics into computable ones and skipped ones.

    Pure and dependency-free, so the gating logic is unit-tested without RAGAS
    installed or any model loaded.
    """
    has_gt = bool(samples) and all(s.ground_truth for s in samples)
    usable: list[str] = []
    skipped: dict[str, str] = {}
    for m in requested:
        if m in NEEDS_GROUND_TRUTH and not has_gt:
            skipped[m] = "needs a reference answer for every sample (expert gold-set)"
        else:
            usable.append(m)
    return usable, skipped


def to_ragas_dataset(samples: list[RAGEvalSample]) -> dict[str, list]:
    """Column-wise dict in the shape RAGAS expects."""
    return {
        "question": [s.query for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [list(s.contexts) for s in samples],
        "ground_truth": [s.ground_truth or "" for s in samples],
    }


def run_ragas_evaluation(
    samples: list[RAGEvalSample],
    config: EvalConfig,
    output_path: Path | None = None,
    *,
    judge_llm: Any | None = None,
    judge_embeddings: Any | None = None,
) -> RAGASScores:
    """Compute RAGAS scores over ``samples``.

    Parameters
    ----------
    samples
        Evaluation set.
    config
        Provides ``config.ragas.metrics`` (the subset to compute).
    output_path
        Write the per-sample table here as CSV.
    judge_llm, judge_embeddings
        Models RAGAS should judge with. **Pass the locally-loaded Llama and
        embedder** to keep the run free and offline; leaving them ``None`` uses
        whatever RAGAS defaults to, which may incur API cost.

    Returns
    -------
    RAGASScores
        Metrics that could not be computed are ``None`` and listed in
        ``skipped``. If ``ragas`` is not installed the call does not raise —
        it returns empty scores with the reason recorded, so an evaluation run
        degrades instead of dying.
    """
    requested = list(config.ragas.metrics)
    usable, skipped = select_metrics(requested, samples)

    if not samples:
        return RAGASScores(skipped={**skipped, "_all": "no samples supplied"})
    if not usable:
        return RAGASScores(skipped=skipped)

    if judge_llm is None:
        _LOGGER.warning(
            "run_ragas_evaluation: no judge_llm supplied — RAGAS will use its "
            "own default, which may call a paid API. Pass the local Llama."
        )

    try:
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas import metrics as ragas_metrics  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - optional dependency
        _LOGGER.warning("RAGAS unavailable (%s); skipping RAGAS metrics.", exc)
        return RAGASScores(skipped={**skipped, "_all": f"ragas not available: {exc}"})

    metric_objs = []
    for name in usable:
        obj = getattr(ragas_metrics, name, None)
        if obj is None:
            skipped[name] = "not provided by the installed ragas version"
        else:
            metric_objs.append(obj)
    if not metric_objs:
        return RAGASScores(skipped=skipped)

    kwargs: dict[str, Any] = {}
    if judge_llm is not None:
        kwargs["llm"] = judge_llm
    if judge_embeddings is not None:
        kwargs["embeddings"] = judge_embeddings

    result = ragas_evaluate(Dataset.from_dict(to_ragas_dataset(samples)),
                            metrics=metric_objs, **kwargs)

    scores = RAGASScores(skipped=skipped)
    for name in usable:
        value = None
        try:
            value = float(result[name])
        except Exception:  # noqa: BLE001 - metric absent from the result
            skipped.setdefault(name, "not returned by ragas")
        if hasattr(scores, name):
            setattr(scores, name, value)

    if output_path is not None:
        _write_csv(result, Path(output_path))
    return scores


def _write_csv(result: Any, path: Path) -> None:
    """Best-effort per-sample dump; never fails the evaluation run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_pandas().to_csv(path, index=False)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Could not write RAGAS per-sample CSV to %s: %s", path, exc)


__all__ = [
    "NEEDS_GROUND_TRUTH",
    "RAGASScores",
    "RAGEvalSample",
    "run_ragas_evaluation",
    "select_metrics",
    "to_ragas_dataset",
]
