"""Tests for the Phase 11 query set + its book/passage-level scoring."""

from __future__ import annotations

import json

import pytest

from src.eval.query_set import (
    DEFAULT_QUERY_SET,
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
    """Honest-refusal controls must exist, and carry no relevant books."""
    cases = load_query_set()
    negatives = [c for c in cases
                 if not c.expect_answerable and c.unanswerable_kind != "no_coverage"]
    assert negatives, "need cases where refusing is the correct behaviour"
    for n in negatives:
        assert n.relevant_books == []
        assert n.note, "a negative should say why it is unanswerable"


def test_no_coverage_cases_keep_their_book_labels() -> None:
    """The §6m coverage check marked 13 disease queries unanswerable because no
    passage matches the lesion description — NOT because the books are off-topic.
    They keep `relevant_books` (retrieval is still scored on them) but must not be
    counted as answerable when reporting over-refusal."""
    cases = load_query_set()
    gap = [c for c in cases if c.unanswerable_kind == "no_coverage"]
    assert gap, "expected the coverage-gap cases from EXPERIMENT_LOG.md §6m"
    for c in gap:
        assert not c.expect_answerable
        assert c.relevant_books, "a coverage-gap case is still topically in scope"
        assert c.note, "record why the corpus cannot answer it"
        assert c.top1_rerank_score is not None, "record the measured score either way"


def test_no_coverage_is_justified_by_score_or_by_reading() -> None:
    """A case may be unanswerable on either kind of evidence, not only a low score.

    Until §6n every ``no_coverage`` label came from a cross-encoder score below 0.15,
    and this test asserted that. §6n showed the score is sound evidence of ABSENCE but
    not of PRESENCE, because it measures vocabulary overlap: n16 ("protecting harvested
    grain from insects during storage") scored **0.6254** yet its top passages were a
    materials table about roots and branches and a description of threshing. Reading
    them is what settled it.

    So the requirement is evidence of one kind or the other — a score below the floor,
    or a recorded reading grade — never an unjustified label.
    """
    cases = load_query_set()
    gap = [c for c in cases if c.unanswerable_kind == "no_coverage"]
    assert gap
    for c in gap:
        low_score = c.top1_rerank_score is not None and c.top1_rerank_score < 0.15
        read_as_no = getattr(c, "human_grade", "") == "no"
        assert low_score or read_as_no, (
            f"{c.id} is marked no_coverage on neither evidence: "
            f"score={c.top1_rerank_score}, grade={getattr(c, 'human_grade', None)!r}"
        )


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


# --------------------------------------------------------------------------- #
# query_source: authored wording vs what the bridge really generates (§6o).
#
# The deployed pipeline is photo -> disease label -> Strategy B -> retrieval. It never
# emits an authored query, so every number measured on authored wording describes a
# system that does not exist. §6o measured the difference: the bridge scored 3.8x worse
# than the authored ceiling before its prompt was fixed, and level with it after.
# --------------------------------------------------------------------------- #


def test_generated_source_swaps_in_the_bridges_real_wording() -> None:
    authored = {c.id: c for c in load_query_set()}
    generated = {c.id: c for c in load_query_set(query_source="generated")}
    swapped = [i for i in authored if authored[i].query != generated[i].query]
    assert swapped, "no case was swapped; the generated_query field is missing"
    for i in swapped:
        assert generated[i].query == authored[i].generated_query


def test_cases_without_a_generated_form_keep_their_authored_wording() -> None:
    """Domain queries and the negative controls cannot come from the vision pipeline,
    so there is nothing to substitute and they must pass through untouched."""
    for c in load_query_set(query_source="generated"):
        if not c.generated_query:
            assert c.query == c.authored_query


def test_authored_wording_survives_the_swap() -> None:
    """Both must be reportable from one load: the generated wording is the system, the
    authored wording is the ceiling it is measured against."""
    for c in load_query_set(query_source="generated"):
        assert c.authored_query, f"{c.id} lost its authored wording"


def test_authored_is_the_default_so_old_results_stay_reproducible() -> None:
    assert [c.query for c in load_query_set()] == \
           [c.query for c in load_query_set(query_source="authored")]


def test_unknown_query_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="authored"):
        load_query_set(query_source="llama")


def test_every_disease_query_has_a_generated_form() -> None:
    """A disease case with no recorded bridge output would silently fall back to
    authored wording and quietly inflate a 'real system' run."""
    missing = [c.id for c in load_query_set() if c.disease and not c.generated_query]
    assert not missing, f"disease queries with no recorded Strategy-B output: {missing}"


def test_relevant_books_are_not_copied_from_retrieval_output() -> None:
    """Labels must be authored independently, or Precision@5 grades retrieval against
    its own output.

    The 16 domain queries added 2026-09-26 originally took relevant_books from the top-3
    books of their own retrieval run. Every retrieval variant was inflated; keyword_only
    P@5 rising 0.33 -> 0.53 purely from adding queries was the visible symptom. They were
    re-labelled from each treatise's subject matter, and each must record the basis so
    the provenance of the label is auditable rather than assumed.
    """
    added = [q for q in json.loads(DEFAULT_QUERY_SET.read_text(encoding="utf-8"))["queries"]
             if q.get("added") == "2026-09-26"]
    assert added, "expected the domain queries added on 2026-09-26"
    for q in added:
        assert q.get("relevant_books"), f"{q['id']} has no relevant_books"
        assert q.get("relevant_books_basis"), (
            f"{q['id']} has no recorded basis for its relevant_books — an unexplained "
            "label cannot be distinguished from one copied out of a ranking"
        )


def test_every_corpus_book_is_a_relevant_label_somewhere() -> None:
    """A book in the corpus that appears in no query's relevant_books is scored as
    irrelevant every time it is retrieved. That silently depressed every retrieval
    variant twice: the new books in 6j, and Vishvavallabha on 2026-09-28. Adding a book
    to the corpus must come with labelling it."""
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "configs/corpus/books.yaml").read_text(encoding="utf-8"))
    built = {b["id"] for b in cfg["books"]
             if b.get("status") in ("ready", "ready_external")
             and (root / "corpus/chunks" / f"{b['id']}.jsonl").is_file()
             and (root / "corpus/chunks" / f"{b['id']}.jsonl").stat().st_size > 0}
    labelled = {b for c in load_query_set() for b in c.relevant_books}
    missing = sorted(built - labelled)
    assert not missing, f"books in the corpus but in no query's relevant_books: {missing}"
