"""Load the Phase 11 evaluation query set and turn retrievals into IR scores.

The query set (``data/eval/silver_queries.json``) is SILVER: authored by the
project, not yet expert-validated, so anything computed from it is preliminary
until the expert gold-set ratifies or replaces the labels.

Two labelling levels are supported, and the distinction matters:

* **book level** (available now) — ``relevant_books`` says which treatise(s)
  plausibly address the condition. A retrieved passage counts as relevant if it
  came from one of them. This supports Precision@k, nDCG, MRR and Hit@k.
* **passage level** (later) — ``relevant_chunk_ids`` names the specific passages.
  Only then is **Recall@k** meaningful, because only then do we know how many
  relevant passages exist in total. Book-level scoring therefore reports Recall
  as ``None`` rather than printing a number that looks real but isn't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.eval.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_QUERY_SET = Path(__file__).resolve().parent.parent.parent / "data/eval/silver_queries.json"


@dataclass
class QueryCase:
    """One evaluation query plus its relevance labels."""

    id: str
    query: str
    crop: str = ""
    disease: str | None = None
    relevant_books: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expect_answerable: bool = True
    note: str = ""

    @property
    def has_passage_labels(self) -> bool:
        """True once specific passages are labelled (enables Recall@k)."""
        return bool(self.relevant_chunk_ids)


def load_query_set(path: Path | str | None = None) -> list[QueryCase]:
    """Read the query set JSON into :class:`QueryCase` objects."""
    p = Path(path) if path is not None else DEFAULT_QUERY_SET
    payload = json.loads(p.read_text(encoding="utf-8"))
    return [
        QueryCase(
            id=q["id"], query=q["query"], crop=q.get("crop", ""),
            disease=q.get("disease"),
            relevant_books=list(q.get("relevant_books") or []),
            relevant_chunk_ids=list(q.get("relevant_chunk_ids") or []),
            expect_answerable=bool(q.get("expect_answerable", True)),
            note=q.get("note", ""),
        )
        for q in payload["queries"]
    ]


def answerable_cases(cases: list[QueryCase]) -> list[QueryCase]:
    """Only the queries the corpus is expected to answer.

    The deliberate negatives are scored separately: for those, success means an
    honest refusal, not a good ranking.
    """
    return [c for c in cases if c.expect_answerable]


def _book_of(chunk: tuple[str, str]) -> str:
    return chunk[1]


def score_case(
    case: QueryCase,
    ranked_chunks: list[tuple[str, str]],
    k: int = 5,
) -> dict[str, float | None | str]:
    """Score one query's ranking.

    Parameters
    ----------
    case
        The query and its labels.
    ranked_chunks
        Retrieved chunks in rank order as ``(chunk_id, book_id)`` pairs.
    k
        Rank cut-off.

    Returns a per-query dict. ``recall`` is ``None`` under book-level labels —
    see the module docstring for why reporting a number there would mislead.
    """
    ranked_ids = [cid for cid, _ in ranked_chunks]

    if case.has_passage_labels:
        relevant = set(case.relevant_chunk_ids)
        recall: float | None = recall_at_k(ranked_ids, relevant, k)
        mode = "passage"
    else:
        wanted = {b.lower() for b in case.relevant_books}
        # A retrieved passage is relevant if it came from a plausibly-relevant
        # treatise. Recall is left undefined: the corpus-wide count of relevant
        # passages is unknown without passage labels.
        relevant = {cid for cid, book in ranked_chunks if str(book).lower() in wanted}
        recall = None
        mode = "book"

    return {
        "id": case.id, "mode": mode,
        "precision": precision_at_k(ranked_ids, relevant, k),
        "recall": recall,
        "ndcg": ndcg_at_k(ranked_ids, relevant, k),
        "rr": reciprocal_rank(ranked_ids, relevant),
        "hit": hit_at_k(ranked_ids, relevant, k),
    }


def aggregate(rows: list[dict], k: int = 5) -> dict[str, float | None | int]:
    """Mean the per-query rows into a single reportable result."""
    if not rows:
        return {"k": k, "n": 0, "precision": 0.0, "recall": None,
                "ndcg": 0.0, "mrr": 0.0, "hit": 0.0}

    def _mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    recalls = [r["recall"] for r in rows if r.get("recall") is not None]
    return {
        "k": k, "n": len(rows),
        "precision": _mean("precision"),
        # only reported once every scored query carries passage labels
        "recall": (sum(recalls) / len(recalls)) if len(recalls) == len(rows) else None,
        "ndcg": _mean("ndcg"),
        "mrr": _mean("rr"),
        "hit": _mean("hit"),
    }


__all__ = [
    "DEFAULT_QUERY_SET",
    "QueryCase",
    "aggregate",
    "answerable_cases",
    "load_query_set",
    "score_case",
]
