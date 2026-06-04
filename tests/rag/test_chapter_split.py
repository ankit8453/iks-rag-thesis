"""Tests for :mod:`src.rag.corpus.chapter_split`."""

from __future__ import annotations

import pytest

from src.rag.corpus.chapter_split import ChapterSpan, locate_chapters, to_roman


def test_to_roman_basic_cases() -> None:
    assert to_roman(1) == "I"
    assert to_roman(4) == "IV"
    assert to_roman(9) == "IX"
    assert to_roman(40) == "XL"
    assert to_roman(54) == "LIV"
    assert to_roman(55) == "LV"
    assert to_roman(57) == "LVII"


def test_locate_chapters_finds_heading_independent_of_page_offset() -> None:
    """The page-offset-proof property: putting an offset of garbage at the
    front of the corpus must not change WHICH page chapters are found on."""
    chapter_titles = {
        21: "Pregnancy of Clouds",
        24: "Conjunction with Rohini",
        55: "Treatment of Trees",
    }

    # Without offset.
    pages_a = [
        "front matter unrelated content",
        "Chapter XXI — Pregnancy of Clouds\nVerse 1 lorem ipsum",
        "more clouds prose",
        "Chapter XXIV — Conjunction with Rohini\nVerse content",
        "Chapter LV — Treatment of Trees\nDiseased trees should...",
        "trailing content",
    ]
    spans_a = locate_chapters(pages_a, chapter_titles)
    assert set(spans_a.keys()) == {21, 24, 55}
    assert spans_a[21].start_page_idx == 1
    assert spans_a[24].start_page_idx == 3
    assert spans_a[55].start_page_idx == 4

    # With a 5-page-offset prefix of garbage. Chapter pages must shift
    # by +5, NOT stay at the absolute page numbers.
    offset_pages = ["garbage filler page"] * 5
    pages_b = offset_pages + pages_a
    spans_b = locate_chapters(pages_b, chapter_titles)
    assert spans_b[21].start_page_idx == 6
    assert spans_b[24].start_page_idx == 8
    assert spans_b[55].start_page_idx == 9


def test_locate_chapters_matches_roman_only_form() -> None:
    """A heading with only a Roman-numeral marker (no title text) still scores."""
    chapter_titles = {40: "Growth of Crops"}
    pages = [
        "intro",
        "Chapter XL — \nFollowed by crop prose without the title spelled out.",
        "more text",
    ]
    spans = locate_chapters(pages, chapter_titles)
    assert 40 in spans
    assert spans[40].start_page_idx == 1


def test_locate_chapters_matches_title_only_form() -> None:
    """A running header like ``Treatment of Trees LV 533`` still scores."""
    chapter_titles = {55: "Treatment of Trees"}
    pages = [
        "intro",
        "Treatment of Trees LV 533\nthe diseased tree should be...",
    ]
    spans = locate_chapters(pages, chapter_titles)
    assert 55 in spans
    assert spans[55].start_page_idx == 1


def test_locate_chapters_returns_half_open_spans_in_document_order() -> None:
    chapter_titles = {
        21: "Pregnancy of Clouds",
        22: "Retention of Embryo",
    }
    pages = [
        "intro",
        "Chapter XXI — Pregnancy of Clouds\nA",
        "more A",
        "Chapter XXII — Retention of Embryo\nB",
        "more B",
        "more B again",
    ]
    spans = locate_chapters(pages, chapter_titles)
    assert spans[21].end_page_idx == spans[22].start_page_idx  # contiguous
    assert spans[22].end_page_idx == len(pages)               # last chapter runs to EOF


def test_locate_chapters_warns_but_does_not_crash_when_chapter_missing(caplog) -> None:
    chapter_titles = {
        21: "Pregnancy of Clouds",
        99: "Nonexistent Title That Never Appears",
    }
    pages = ["intro", "Chapter XXI — Pregnancy of Clouds\nA"]
    spans = locate_chapters(pages, chapter_titles)
    assert 21 in spans
    assert 99 not in spans          # not located, omitted (per spec)
    assert "NOT FOUND" in caplog.text


def test_locate_chapters_empty_pages_returns_empty() -> None:
    assert locate_chapters([], {1: "Anything"}) == {}
