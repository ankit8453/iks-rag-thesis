"""Information-retrieval metrics for the Phase 11 evaluation.

Pure, dependency-free ranking metrics computed from two things per query:
the ranked list of retrieved chunk IDs, and the set of IDs judged relevant.

These are the standard IR measures the supervisor asked for — Precision@k,
Recall@k, and nDCG — plus MRR and Hit@k. They contain no model, no network and
no LLM, so they run and unit-test locally; the relevance labels come from the
(silver, then expert gold) query set. Binary relevance is assumed: a chunk is
either relevant to a query or not, which matches how the query set is labelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    hits = sum(1 for cid in topk if cid in relevant)
    return hits / len(topk)


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant chunks found within the top-k."""
    if not relevant:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & relevant) / len(relevant)


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant chunk is in the top-k, else 0.0."""
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant chunk (0 if none retrieved)."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain at k (binary gains)."""
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Normalised DCG at k — DCG divided by the ideal DCG for this query."""
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant, k) / idcg


@dataclass
class RetrievalMetrics:
    """Mean IR scores across a set of queries, at a fixed cut-off ``k``."""

    k: int
    n_queries: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    hit_at_k: float
    per_query: list[dict[str, float]] = field(default_factory=list)

    def as_row(self) -> dict[str, float]:
        """Flat dict for a results table / CSV."""
        return {
            "k": self.k, "n": self.n_queries,
            f"P@{self.k}": round(self.precision_at_k, 4),
            f"R@{self.k}": round(self.recall_at_k, 4),
            f"nDCG@{self.k}": round(self.ndcg_at_k, 4),
            "MRR": round(self.mrr, 4),
            f"Hit@{self.k}": round(self.hit_at_k, 4),
        }


def evaluate_retrieval(
    results: list[tuple[list[str], set[str]]],
    k: int = 5,
) -> RetrievalMetrics:
    """Aggregate IR metrics over many queries.

    Parameters
    ----------
    results
        One ``(ranked_chunk_ids, relevant_chunk_ids)`` pair per query.
    k
        Rank cut-off for the @k metrics.

    Queries with no relevant chunks labelled are skipped (they cannot inform
    precision/recall) and the mean is taken over the remaining queries.
    """
    per_query: list[dict[str, float]] = []
    for ranked, relevant in results:
        if not relevant:
            continue
        per_query.append({
            "precision": precision_at_k(ranked, relevant, k),
            "recall": recall_at_k(ranked, relevant, k),
            "ndcg": ndcg_at_k(ranked, relevant, k),
            "rr": reciprocal_rank(ranked, relevant),
            "hit": hit_at_k(ranked, relevant, k),
        })

    n = len(per_query)
    mean = (lambda key: sum(q[key] for q in per_query) / n) if n else (lambda key: 0.0)
    return RetrievalMetrics(
        k=k, n_queries=n,
        precision_at_k=mean("precision"), recall_at_k=mean("recall"),
        ndcg_at_k=mean("ndcg"), mrr=mean("rr"), hit_at_k=mean("hit"),
        per_query=per_query,
    )


__all__ = [
    "RetrievalMetrics",
    "dcg_at_k",
    "evaluate_retrieval",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
