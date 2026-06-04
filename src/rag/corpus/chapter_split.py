"""Locate chapters by English heading text, ignoring PDF page numbers.

The Brihat Samhita PDF used in Phase 3 has a PDF-page <-> printed-page
offset that **drifts** as you go deeper into the volume (~8 pages near
the front, ~45 near the end). That makes any chapter-by-page-index
scheme unsafe. Instead this module scans the OCR output for English
heading strings and Roman-numeral chapter markers.

Detected forms (case-insensitive, OCR-noise-tolerant):

- ``Chapter XXIV — Conjunction with Rohini``
- ``CHAPTER LV - Treatment of Trees``
- ``Treatment of Trees LV 533``  (a running header — its title is still useful)
- ``Conjunction with Rohini  XXIV``
- The bare title alone: ``Treatment of Trees``

Roman-numeral helpers handle I..LXXXVIII (chapters 1..88), enough for
Brihat Samhita Part 1's 57 chapters and any future book we'd realistically
add to the §15 corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class ChapterSpan:
    """A located chapter: 0-based ``[start_page_idx, end_page_idx)`` half-open."""

    chapter_number: int
    title: str
    start_page_idx: int
    end_page_idx: int


_ROMAN_NUMERAL_MAP = {
    "M": 1000, "CM": 900, "D": 500, "CD": 400,
    "C": 100, "XC": 90, "L": 50, "XL": 40,
    "X": 10, "IX": 9, "V": 5, "IV": 4, "I": 1,
}


def to_roman(n: int) -> str:
    """Standard 1..3999 integer-to-Roman conversion."""
    if n <= 0 or n >= 4000:
        raise ValueError(f"Roman numerals only handle 1..3999; got {n}")
    out: list[str] = []
    for symbol, value in _ROMAN_NUMERAL_MAP.items():
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def _title_regex(title: str) -> re.Pattern[str]:
    """Build a fuzzy regex for an expected chapter title.

    - Case-insensitive.
    - Internal whitespace runs match any whitespace (handles OCR breaks
      mid-title).
    - Word boundaries on both ends.
    """
    parts = [re.escape(word) for word in title.split()]
    body = r"\s+".join(parts)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def _roman_chapter_regex(chapter_number: int) -> re.Pattern[str]:
    """Match ``Chapter LXIII`` / ``CHAPTER LXIII`` / ``Chapter — LXIII`` forms."""
    roman = to_roman(chapter_number)
    return re.compile(
        rf"\bchapter\b[\s\W]{{0,8}}{roman}\b",
        re.IGNORECASE,
    )


def _heading_score(
    page_text: str,
    *,
    chapter_number: int,
    title_re: re.Pattern[str],
    roman_re: re.Pattern[str],
) -> int:
    """Heuristic score for "this page is where chapter N begins".

    Score = 2 if both Roman-numeral chapter marker AND title appear,
            1 if only the title appears (e.g. a running header), and
            1 if only the Roman marker appears, else 0.
    Returns the score; the caller picks the highest-scoring page.
    """
    has_title = bool(title_re.search(page_text))
    has_roman = bool(roman_re.search(page_text))
    if has_title and has_roman:
        return 2
    if has_title or has_roman:
        return 1
    return 0


def locate_chapters(
    pages: list[str],
    chapter_titles: dict[int, str],
) -> dict[int, ChapterSpan]:
    """Find the page span of each wanted chapter by heading scan.

    Parameters
    ----------
    pages
        Cleaned per-page text in document order. ``pages[i]`` is the
        text of the (i+1)-th OCR page.
    chapter_titles
        ``{chapter_number: english_title}`` for the chapters of interest.

    Returns
    -------
    dict[int, ChapterSpan]
        One entry per chapter that was located. Chapters that could
        NOT be located are logged at WARNING level and OMITTED from
        the dict — the caller decides whether that's acceptable.
    """
    if not pages:
        return {}

    located: dict[int, int] = {}   # chapter_number -> best start page idx
    for chapter_number, title in chapter_titles.items():
        title_re = _title_regex(title)
        roman_re = _roman_chapter_regex(chapter_number)

        # Find the FIRST page that scores >= 1, preferring score 2.
        best_idx = -1
        best_score = 0
        for idx, page_text in enumerate(pages):
            score = _heading_score(
                page_text,
                chapter_number=chapter_number,
                title_re=title_re,
                roman_re=roman_re,
            )
            if score > best_score:
                best_score = score
                best_idx = idx
                if score == 2:
                    break  # can't beat a perfect match — earliest wins.

        if best_idx < 0:
            _LOGGER.warning(
                "locate_chapters: chapter %d (%r) NOT FOUND in any of %d pages",
                chapter_number, title, len(pages),
            )
            continue
        located[chapter_number] = best_idx

    # Sort by page index and convert to [start, end) spans.
    sorted_chapters = sorted(located.items(), key=lambda kv: kv[1])
    spans: dict[int, ChapterSpan] = {}
    for i, (chapter_number, start_idx) in enumerate(sorted_chapters):
        if i + 1 < len(sorted_chapters):
            end_idx = sorted_chapters[i + 1][1]
        else:
            end_idx = len(pages)
        spans[chapter_number] = ChapterSpan(
            chapter_number=chapter_number,
            title=chapter_titles[chapter_number],
            start_page_idx=start_idx,
            end_page_idx=end_idx,
        )
        _LOGGER.info(
            "locate_chapters: chapter %d (%r) at pages %d..%d",
            chapter_number, chapter_titles[chapter_number],
            start_idx + 1, end_idx,
        )
    return spans


__all__ = [
    "ChapterSpan",
    "locate_chapters",
    "to_roman",
]
