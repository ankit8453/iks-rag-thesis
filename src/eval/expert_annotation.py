"""Expert annotation IO.

Expert reviewers (domain agronomists) fill a CSV with one row per
``(query, answer)`` pair. The schema below is what
:func:`load_annotations` validates against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExpertAnnotation:
    """One expert annotation row.

    Attributes
    ----------
    query_id : str
    annotator_id : str
    correctness : int
        1-5 Likert score on factual correctness.
    groundedness : int
        1-5 Likert score on grounding in classical texts.
    safety : Literal["safe", "neutral", "unsafe"]
        Categorical safety label. ``unsafe`` flags recommendations that
        could harm the crop / farmer.
    notes : str
        Free-form reviewer comment.
    """

    query_id: str
    annotator_id: str
    correctness: int
    groundedness: int
    safety: Literal["safe", "neutral", "unsafe"]
    notes: str


def load_annotations(csv_path: Path) -> list[ExpertAnnotation]:
    """Load and validate the expert annotation CSV.

    Parameters
    ----------
    csv_path : Path

    Returns
    -------
    list[ExpertAnnotation]

    Raises
    ------
    NotImplementedError
        Phase 9 — Week 30.
    """
    raise NotImplementedError("Phase 9 — Week 30: implement expert annotation IO.")
