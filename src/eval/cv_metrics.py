"""Classification metric reporting.

Per supervisor guardrail #3, every evaluation report must include
per-class precision/recall/F1 and a confusion matrix — not just headline
accuracy. :class:`ClassificationReport` enforces that shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerClassMetrics:
    """Precision / recall / F1 / support for a single class."""

    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationReport:
    """Full classification report.

    Attributes
    ----------
    accuracy : float
        Overall accuracy (kept for sanity-checking but never reported in
        isolation).
    macro_f1 : float
        Unweighted mean of per-class F1.
    weighted_f1 : float
        Support-weighted mean of per-class F1.
    per_class : list[PerClassMetrics]
        Per-class metrics. Always populated — see guardrail #3.
    confusion_matrix : list[list[int]]
        Square matrix ``cm[true][pred]`` of counts.
    class_names : list[str]
        Order matches rows/cols of ``confusion_matrix``.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: list[PerClassMetrics] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)


def compute_per_class_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> ClassificationReport:
    """Compute per-class precision/recall/F1 plus a confusion matrix.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth class indices.
    y_pred : list[int]
        Predicted class indices.
    class_names : list[str]
        Index-aligned class names.

    Returns
    -------
    ClassificationReport
        With all per-class fields populated.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 5 — Week 18 (CV evaluation).
    """
    raise NotImplementedError("Phase 5 — Week 18: implement per-class metrics.")
