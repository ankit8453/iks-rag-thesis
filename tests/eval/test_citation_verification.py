"""Tests for src.eval.citation_verification."""

from __future__ import annotations

from src.eval.citation_verification import (
    extract_citations,
    verify_citations_in_context,
)


def test_extract_citations_finds_chunk_ids() -> None:
    answer = "Apply neem oil [vrikshayurveda:42] and water sparingly [krishi_parashara:7]."
    assert extract_citations(answer) == ["vrikshayurveda:42", "krishi_parashara:7"]


def test_extract_citations_preserves_duplicates() -> None:
    answer = "See [a:1]. Also see [a:1] again."
    assert extract_citations(answer) == ["a:1", "a:1"]


def test_extract_citations_empty_answer() -> None:
    assert extract_citations("No citations in this answer.") == []


def test_verify_citations_marks_invalid() -> None:
    answer = "Cite [valid:1] and also [fabricated:99]."
    report = verify_citations_in_context(answer, retrieved_chunk_ids=["valid:1", "other:2"])
    assert "valid:1" in report.valid_ids
    assert "fabricated:99" in report.invalid_ids


def test_verify_citations_coverage_is_zero_when_nothing_cited() -> None:
    report = verify_citations_in_context("Plain answer.", retrieved_chunk_ids=["a:1", "b:2"])
    assert report.coverage == 0.0
    assert report.cited_ids == []


def test_verify_citations_coverage_one_when_all_cited() -> None:
    answer = "Use [a:1] and [b:2]."
    report = verify_citations_in_context(answer, retrieved_chunk_ids=["a:1", "b:2"])
    assert report.coverage == 1.0
