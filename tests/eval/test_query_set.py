"""Tests for the Phase 11 query set + its book/passage-level scoring."""

from __future__ import annotations

import pytest

from src.eval.query_set import (
    QueryCase,
    aggregate,
    answerable_cases,
    load_query_set,
    score_case,
)


def test_query_set_loads_and_is_symptom_led() -> None:
    cases = load_query_set()
    assert len(cases) >= 20
    ids = [c.id for c in cases]
    assert len(set(ids)) == len(ids), "query ids must be unique"
    # queries describe SYMPTOMS, so the crop name should not lead the text
    for c in cases:
        if c.crop and c.crop != "general":
            assert not c.query.lower().startswith(c.crop.lower())


def test_query_set_has_deliberate_negatives() -> None:
    """Honest-refusal cases must exist, and carry no relevant books."""
    cases = load_query_set()
    negatives = [c for c in cases if not c.expect_answerable]
    assert negatives, "need cases where refusing is the correct behaviour"
    for n in negatives:
        assert n.relevant_books == []
        assert n.note, "a negative should say why it is unanswerable"


def test_answerable_cases_filters_negatives() -> None:
    cases = load_query_set()
    pos = answerable_cases(cases)
    assert len(pos) == len([c for c in cases if c.expect_answerable])
    assert all(c.expect_answerable for c in pos)


def test_book_level_scoring_marks_in_book_hits_relevant() -> None:
    case = QueryCase(id="t1", query="white powdery coating",
                     relevant_books=["vrikshayurveda"])
    ranked = [("c1", "vrikshayurveda"), ("c2", "krishi_parashara"),
              ("c3", "vrikshayurveda")]
    row = score_case(case, ranked, k=3)
    assert row["mode"] == "book"
    assert row["precision"] == pytest.approx(2 / 3)   # c1, c3 in-book
    assert row["hit"] == 1.0
    assert row["rr"] == pytest.approx(1.0)            # first hit at rank 1


def test_book_level_recall_is_withheld_not_faked() -> None:
    """Recall needs the true relevant count — unknown without passage labels."""
    case = QueryCase(id="t2", query="q", relevant_books=["vrikshayurveda"])
    row = score_case(case, [("c1", "vrikshayurveda")], k=5)
    assert row["recall"] is None


def test_passage_level_labels_enable_recall() -> None:
    case = QueryCase(id="t3", query="q", relevant_books=["vrikshayurveda"],
                     relevant_chunk_ids=["c1", "c9"])
    row = score_case(case, [("c1", "vrikshayurveda"), ("c2", "upavanavinoda")], k=5)
    assert row["mode"] == "passage"
    assert row["recall"] == pytest.approx(0.5)        # found c1 of {c1,c9}


def test_aggregate_withholds_recall_unless_all_rows_have_it() -> None:
    mixed = [
        {"precision": 1.0, "recall": 0.5, "ndcg": 1.0, "rr": 1.0, "hit": 1.0},
        {"precision": 0.0, "recall": None, "ndcg": 0.0, "rr": 0.0, "hit": 0.0},
    ]
    assert aggregate(mixed)["recall"] is None

    full = [
        {"precision": 1.0, "recall": 1.0, "ndcg": 1.0, "rr": 1.0, "hit": 1.0},
        {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "rr": 0.0, "hit": 0.0},
    ]
    agg = aggregate(full)
    assert agg["recall"] == pytest.approx(0.5)
    assert agg["precision"] == pytest.approx(0.5)
    assert agg["n"] == 2


def test_aggregate_handles_no_rows() -> None:
    assert aggregate([])["n"] == 0
