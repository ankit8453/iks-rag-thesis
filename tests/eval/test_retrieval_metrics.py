"""Tests for the Phase 11 IR metrics — checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from src.eval.retrieval_metrics import (
    evaluate_retrieval,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# a ranking with relevant chunks at positions 1 and 3 (1-indexed)
RANKED = ["a", "b", "c", "d", "e"]
REL = {"a", "c"}


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED, REL, 1) == pytest.approx(1.0)     # a
    assert precision_at_k(RANKED, REL, 2) == pytest.approx(0.5)     # a,b
    assert precision_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)   # a,b,c
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(0.4)     # 2 of 5


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, REL, 1) == pytest.approx(0.5)   # found a of {a,c}
    assert recall_at_k(RANKED, REL, 3) == pytest.approx(1.0)   # found both
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(1.0)


def test_hit_and_reciprocal_rank() -> None:
    assert hit_at_k(RANKED, REL, 1) == 1.0
    assert hit_at_k(["x", "y"], REL, 5) == 0.0
    assert reciprocal_rank(RANKED, REL) == pytest.approx(1.0)          # first hit at rank 1
    assert reciprocal_rank(["x", "a"], REL) == pytest.approx(0.5)      # first hit at rank 2
    assert reciprocal_rank(["x", "y"], REL) == pytest.approx(0.0)      # no hit


def test_ndcg_matches_hand_computation() -> None:
    # hits at ranks 1 and 3 -> DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # ideal (both at ranks 1,2) -> IDCG = 1 + 1/log2(3) = 1 + 0.6309 = 1.6309
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(expected, abs=1e-6)


def test_perfect_ranking_scores_one() -> None:
    ranked = ["a", "c", "b", "d"]
    assert precision_at_k(ranked, REL, 2) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, REL, 5) == pytest.approx(1.0)
    assert reciprocal_rank(ranked, REL) == pytest.approx(1.0)


def test_empty_relevant_set_is_zero_not_crash() -> None:
    assert recall_at_k(RANKED, set(), 5) == 0.0
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


def test_evaluate_retrieval_aggregates_and_skips_unlabelled() -> None:
    results = [
        (["a", "b", "c"], {"a"}),        # P@3=1/3, R@3=1, RR=1
        (["x", "y", "z"], {"z"}),        # P@3=1/3, R@3=1, RR=1/3
        (["p", "q"], set()),             # unlabelled -> skipped
    ]
    m = evaluate_retrieval(results, k=3)
    assert m.n_queries == 2                       # third skipped
    assert m.precision_at_k == pytest.approx(1 / 3)
    assert m.recall_at_k == pytest.approx(1.0)
    assert m.mrr == pytest.approx((1.0 + 1 / 3) / 2)
    row = m.as_row()
    assert row["R@3"] == 1.0 and row["n"] == 2
