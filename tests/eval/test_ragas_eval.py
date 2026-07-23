"""Tests for the RAGAS wrapper.

RAGAS itself is not exercised here (it needs a judge model); what is tested is
the gating and degradation logic around it — which metrics are computable
without a reference answer, and that a missing dependency degrades instead of
crashing an evaluation run.
"""

from __future__ import annotations

from src.eval.config import EvalConfig
from src.eval.ragas_eval import (
    NEEDS_GROUND_TRUTH,
    RAGEvalSample,
    run_ragas_evaluation,
    select_metrics,
    to_ragas_dataset,
)


def _sample(gt: str | None = None) -> RAGEvalSample:
    return RAGEvalSample(
        query="white powdery coating on leaves",
        answer="Apply the prescribed paste [Vrikshayurveda, ch.1, v.2].",
        contexts=["a passage about a white coating on foliage"],
        ground_truth=gt,
    )


def test_faithfulness_is_computable_without_a_reference_answer() -> None:
    """The metric we most need does NOT depend on the expert gold-set."""
    usable, skipped = select_metrics(
        ["faithfulness", "answer_relevancy"], [_sample(), _sample()])
    assert usable == ["faithfulness", "answer_relevancy"]
    assert skipped == {}


def test_reference_dependent_metrics_are_skipped_not_zeroed() -> None:
    usable, skipped = select_metrics(
        ["faithfulness", "context_precision", "context_recall"], [_sample()])
    assert usable == ["faithfulness"]
    assert set(skipped) == {"context_precision", "context_recall"}
    for reason in skipped.values():
        assert "reference" in reason


def test_reference_metrics_unlock_when_every_sample_has_ground_truth() -> None:
    samples = [_sample("expert answer"), _sample("expert answer")]
    usable, skipped = select_metrics(["context_precision"], samples)
    assert usable == ["context_precision"]
    assert skipped == {}


def test_partial_ground_truth_still_skips() -> None:
    """One missing reference is enough to make the metric unsound."""
    usable, _ = select_metrics(["context_recall"], [_sample("gt"), _sample(None)])
    assert usable == []


def test_needs_ground_truth_set_is_explicit() -> None:
    assert "faithfulness" not in NEEDS_GROUND_TRUTH
    assert "answer_relevancy" not in NEEDS_GROUND_TRUTH
    assert "context_recall" in NEEDS_GROUND_TRUTH


def test_dataset_shape_matches_ragas_columns() -> None:
    ds = to_ragas_dataset([_sample("gt")])
    assert set(ds) == {"question", "answer", "contexts", "ground_truth"}
    assert ds["contexts"] == [["a passage about a white coating on foliage"]]
    assert ds["ground_truth"] == ["gt"]


def test_missing_ragas_degrades_instead_of_raising(monkeypatch) -> None:
    """An evaluation run must not die because an optional dep is absent."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kw):
        if name.startswith("ragas"):
            raise ImportError("ragas not installed")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    scores = run_ragas_evaluation([_sample()], EvalConfig())
    assert scores.faithfulness is None
    assert "_all" in scores.skipped


def test_no_samples_returns_empty_scores() -> None:
    scores = run_ragas_evaluation([], EvalConfig())
    assert scores.faithfulness is None
    assert scores.skipped
