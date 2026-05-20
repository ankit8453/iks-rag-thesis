"""Smoke tests for the eval module."""

from __future__ import annotations

from src.eval import (
    ClassificationReport,
    EvalConfig,
    compute_per_class_metrics,
    run_ragas_evaluation,
)


def test_eval_config_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.cv.report_per_class is True
    assert cfg.cv.save_confusion_matrix is True
    assert "faithfulness" in cfg.ragas.metrics


def test_classification_report_holds_per_class_and_confusion_matrix() -> None:
    """Guardrail #3: shape must include per_class + confusion_matrix."""
    rep = ClassificationReport(accuracy=0.5, macro_f1=0.5, weighted_f1=0.5)
    assert hasattr(rep, "per_class")
    assert hasattr(rep, "confusion_matrix")
    assert hasattr(rep, "class_names")


def test_compute_per_class_metrics_has_docstring() -> None:
    assert compute_per_class_metrics.__doc__ is not None
    assert "per-class" in (compute_per_class_metrics.__doc__ or "")


def test_run_ragas_evaluation_has_docstring() -> None:
    assert run_ragas_evaluation.__doc__ is not None
