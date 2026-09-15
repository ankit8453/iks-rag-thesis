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


#: A *real* chapter heading, used to find where a chapter ENDS. Three guards, each
#: needed against a false positive seen in the Brihat Samhita OCR:
#:
#: 1. the literal word "chapter" — running headers are ``Treatment of Trees LV 533``
#:    (title + numeral + folio, no "chapter"), so they cannot match;
#: 2. a separator and a capitalised title after the numeral — cross-references such
#:    as ``[Cf. chapter IX]`` or ``as stated in chapter XXI`` have no title and are
#:    rejected. Without this, p.291 (continuation of ch.23 Rainfall) and p.304
#:    (continuation of ch.24 Rohini) were read as chapter starts and truncated
#:    their real chapters;
#: 3. it must appear at the very top of the page (see ``_HEADING_MAX_OFFSET``).
_ANY_CHAPTER_RE = re.compile(
    r"\bchapter\b[\s\W]{0,4}([IVXLCDM]+)\s*[.\-–—:]+\s*[A-Z]",
    re.IGNORECASE,
)

#: How far into the page text a heading may start. Headings are the first thing on
#: the page; a few characters of leading folio/OCR noise are tolerated, but prose
#: mentions further down are not.
_HEADING_MAX_OFFSET = 40

#: Only scan the head of the page at all.
_HEADING_WINDOW = 300


def _from_roman(roman: str) -> int | None:
    """Parse a Roman numeral; ``None`` if it is not a well-formed numeral."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    roman = roman.upper()
    if not roman or any(ch not in values for ch in roman):
        return None
    total = 0
    for i, ch in enumerate(roman):
        v = values[ch]
        nxt = values.get(roman[i + 1]) if i + 1 < len(roman) else None
        total += -v if (nxt is not None and v < nxt) else v
    # round-trip guard: rejects malformed strings such as "IIII" or "VV"
    return total if 0 < total < 4000 and to_roman(total) == roman else None


def find_chapter_starts(pages: list[str]) -> dict[int, int]:
    """Map ``page_idx -> chapter_number`` for every explicit chapter heading found.

    Used to decide where a wanted chapter *ends*. Without this, a wanted chapter's
    span runs to the next **wanted** chapter and silently swallows every chapter in
    between — the defect that put Brihat Samhita chapters XLI-LIII ("Fluctuation of
    Prices", "Signs of Swords", "Architecture", ...) inside chapter XL
    ("Growth of Crops"). See EXPERIMENT_LOG.md §6k.
    """
    starts: dict[int, int] = {}
    for idx, page_text in enumerate(pages):
        head = page_text.lstrip()[:_HEADING_WINDOW]
        m = _ANY_CHAPTER_RE.search(head)
        if not m or m.start() > _HEADING_MAX_OFFSET:
            continue
        number = _from_roman(m.group(1))
        if number is not None:
            starts[idx] = number
    return starts


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

    # Where does EVERY chapter start? A wanted chapter must end at the next chapter
    # heading of any kind, not at the next *wanted* one.
    all_starts = find_chapter_starts(pages)

    # Sort by page index and convert to [start, end) spans.
    sorted_chapters = sorted(located.items(), key=lambda kv: kv[1])
    spans: dict[int, ChapterSpan] = {}
    for i, (chapter_number, start_idx) in enumerate(sorted_chapters):
        # Upper bound: the next wanted chapter (or end of book) — the old behaviour.
        if i + 1 < len(sorted_chapters):
            end_idx = sorted_chapters[i + 1][1]
        else:
            end_idx = len(pages)

        # Tighten it to the next chapter heading that belongs to a DIFFERENT chapter.
        # `min` keeps this a pure narrowing: a span can only shrink, never grow, so a
        # previously-correct span cannot be broken by this step.
        next_any = [p for p, num in all_starts.items()
                    if p > start_idx and num != chapter_number]
        if next_any:
            tightened = min(next_any)
            if tightened < end_idx:
                _LOGGER.info(
                    "locate_chapters: chapter %d end tightened %d -> %d "
                    "(next heading is chapter %d)",
                    chapter_number, end_idx, tightened, all_starts[tightened],
                )
                end_idx = tightened

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
