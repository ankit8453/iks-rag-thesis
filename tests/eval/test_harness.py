"""Tests for the Phase 11 harness — driven entirely by fakes.

The point is to verify the harness's *judgement*: that it rewards a grounded,
citing answer, rewards an honest refusal on an unanswerable query, and penalises
citations that point at passages never retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.eval.baselines import get_variant
from src.eval.harness import (
    format_table,
    run_full_evaluation,
    run_generation_eval,
    run_retrieval_eval,
    run_ungrounded_control,
)
from src.eval.query_set import QueryCase


@dataclass
class _Chunk:
    chunk_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class _Answer:
    answer: str
    retrieved: list


class _Retriever:
    """Returns chunks from a fixed book, so scoring is predictable."""

    def __init__(self, book: str = "vrikshayurveda", n: int = 3):
        self.book, self.n = book, n

    def retrieve(self, query: str, k: int = 5):
        return [_Chunk(f"c{i}", {"book_id": self.book}) for i in range(self.n)]


def _cases() -> list[QueryCase]:
    return [
        QueryCase(id="p1", query="white powdery coating",
                  relevant_books=["vrikshayurveda"], expect_answerable=True),
        QueryCase(id="n1", query="exact NPK percentages", relevant_books=[],
                  expect_answerable=False),
    ]


# ------------------------------------------------------------------ #
# retrieval
# ------------------------------------------------------------------ #


def test_retrieval_run_scores_in_book_hits() -> None:
    cases = [c for c in _cases() if c.expect_answerable]
    run = run_retrieval_eval(cases, _Retriever("vrikshayurveda"), get_variant("full"), k=3)
    assert run.summary["precision"] == pytest.approx(1.0)   # all in-book
    assert run.summary["hit"] == pytest.approx(1.0)
    assert run.variant == "full"


def test_retrieval_run_penalises_off_book_results() -> None:
    cases = [c for c in _cases() if c.expect_answerable]
    run = run_retrieval_eval(cases, _Retriever("krishi_parashara"), get_variant("full"), k=3)
    assert run.summary["precision"] == pytest.approx(0.0)
    assert run.summary["hit"] == pytest.approx(0.0)


def test_retrieval_row_shows_dash_for_recall_without_passage_labels() -> None:
    cases = [c for c in _cases() if c.expect_answerable]
    run = run_retrieval_eval(cases, _Retriever(), get_variant("full"), k=3)
    assert run.as_row(3)["R@3"] == "n/a", "recall must not be faked at book level"


# ------------------------------------------------------------------ #
# generation: grounding + refusal
# ------------------------------------------------------------------ #


class _GoodPipeline:
    """Cites a genuinely retrieved chunk; refuses the unanswerable query."""

    def answer(self, query: str, k: int = 5):
        retrieved = [_Chunk("c1"), _Chunk("c2")]
        if "NPK" in query:
            return _Answer("The retrieved classical-text passages do not contain "
                           "enough information to answer this question.", retrieved)
        return _Answer("Apply the prescribed paste [c1].", retrieved)


class _HallucinatingPipeline:
    """Cites a passage that was never retrieved."""

    def answer(self, query: str, k: int = 5):
        return _Answer("Apply something [not_retrieved_id].", [_Chunk("c1")])


def test_generation_rewards_grounded_answer_and_honest_refusal() -> None:
    run = run_generation_eval(_cases(), _GoodPipeline(), k=5)
    assert run.n_answerable == 1 and run.n_negative == 1
    assert run.grounded_answer_rate == pytest.approx(1.0)
    assert run.valid_citation_rate == pytest.approx(1.0)
    assert run.honest_refusal_rate == pytest.approx(1.0)   # refused the negative
    assert run.over_refusal_rate == pytest.approx(0.0)     # answered the answerable


class _V17Pipeline:
    """The real pipeline shape: cites "[Source, ch.X, v.Y]" and resolves those
    to chunk_ids itself via RAGAnswer.citations / .used_chunk_ids."""

    def answer(self, query: str, k: int = 5):
        ans = _Answer("Apply the paste [Vrikshayurveda, ch.full, v.160.3].",
                      [_Chunk("c1"), _Chunk("c2")])
        ans.citations = ["Vrikshayurveda ch.full v.160.3"]
        ans.used_chunk_ids = ["c1"]
        return ans


def test_v17_style_citations_are_scored_as_grounded() -> None:
    """Regression: the chunk-id regex cannot match "[Source, ch.X, v.Y]", so
    re-parsing the text scored every real answer 0% grounded. The resolved
    fields must be preferred."""
    run = run_generation_eval([_cases()[0]], _V17Pipeline(), k=5)
    assert run.grounded_answer_rate == pytest.approx(1.0)
    assert run.valid_citation_rate == pytest.approx(1.0)
    assert run.per_query[0]["n_valid"] == 1


def test_resolved_citation_to_unretrieved_chunk_is_not_counted() -> None:
    """A resolved id that was never actually retrieved must not count."""

    class _Bogus:
        def answer(self, query, k=5):
            a = _Answer("Apply [X, ch.1, v.1].", [_Chunk("c1")])
            a.citations = ["X ch.1 v.1"]
            a.used_chunk_ids = ["not_retrieved"]
            return a

    run = run_generation_eval([_cases()[0]], _Bogus(), k=5)
    assert run.grounded_answer_rate == pytest.approx(0.0)


def test_generation_penalises_citations_to_unretrieved_passages() -> None:
    run = run_generation_eval([_cases()[0]], _HallucinatingPipeline(), k=5)
    assert run.grounded_answer_rate == pytest.approx(0.0)
    assert run.valid_citation_rate == pytest.approx(0.0)
    assert run.per_query[0]["n_invalid"] == 1


def test_over_refusal_is_detected() -> None:
    """Refusing an answerable query is a failure, not a success."""

    class _AlwaysRefuses:
        def answer(self, query, k=5):
            return _Answer("The retrieved classical-text passages do not contain "
                           "enough information to answer this question.", [_Chunk("c1")])

    run = run_generation_eval([_cases()[0]], _AlwaysRefuses(), k=5)
    assert run.over_refusal_rate == pytest.approx(1.0)
    assert run.grounded_answer_rate == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# ungrounded control + orchestration
# ------------------------------------------------------------------ #


def test_ungrounded_control_gets_no_passages() -> None:
    class _LLM:
        def complete(self, prompt, max_new_tokens=None):
            assert "RETRIEVED" not in prompt.upper()
            return "just use a fungicide"

    out = run_ungrounded_control([_cases()[0]], _LLM())
    assert out["n"] == 1
    assert out["unfounded_citation_rate"] == pytest.approx(0.0)


def test_run_full_evaluation_covers_variants_and_survives_a_failure(monkeypatch) -> None:
    import src.eval.baselines as bmod

    def _build(collection, variant, **kw):
        if variant.name == "dense_only":
            raise RuntimeError("boom")          # one variant fails
        return _Retriever()

    monkeypatch.setattr(bmod, "build_retriever", _build)

    out = run_full_evaluation(_cases(), collection=object(),
                              pipeline=_GoodPipeline(), k=3)
    names = [r.variant for r in out["retrieval"]]
    assert "full" in names and "keyword_only" in names
    assert "dense_only" not in names            # failed variant skipped, run survived
    assert out["n_answerable"] == 1 and out["n_negative"] == 1
    assert out["generation"].honest_refusal_rate == pytest.approx(1.0)
    assert "variant" in out["retrieval_table"]


def test_format_table_renders_rows() -> None:
    txt = format_table([{"variant": "full", "P@5": 0.8}])
    assert "variant" in txt and "full" in txt
    assert len(txt.splitlines()) == 3           # header, separator, one row
