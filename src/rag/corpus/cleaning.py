"""OCR-output cleaning for the Phase 3 IKS corpus.

The PDFs being ingested are translation editions: each English passage
is interleaved with the original Devanagari Sanskrit verse. Tesseract
trained on English transliterates Devanagari into garbage characters,
so we **drop entire lines that are majority Devanagari**
(``U+0900``–``U+097F``) and rely on the English translation only.

We also strip three recurring noise patterns:

1. Running headers/footers — the book title repeats on every page, in
   varying OCR forms (``Brhat Samhita``, ``Brhat Sarhhita``, ``Brihat
   Samhita``, ``Vrikshayurveda``...). A short denylist plus a fuzzy
   match catches them.
2. Standalone page numbers — bare integer lines like ``153``.
3. OCR noise — lines that are essentially punctuation and stray
   accents with fewer than 3 alphabetic characters.

Finally we de-hyphenate line-break splits (``"watering of-"`` +
``"trees"`` → ``"watering of trees"``) and collapse whitespace runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


# Devanagari block (per Unicode 15.0): U+0900..U+097F.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Header / footer denylist: substring match, lower-cased. Anything
# matching these short strings on a line of its own gets dropped. We
# stay conservative; aggressive matching can shave off real content.
_HEADER_DENYLIST = (
    "brhat samhita",
    "brhat sarhhita",
    "brihat samhita",
    "vrikshayurveda",
    "vriksha-ayurveda",
    "vriksha ayurveda",
    "agri-history foundation",
    "asian agri-history foundation",
    "motilal banarsidass",
    "m. ramakrishna bhat",
    "nalini sadhale",
    "surapala",
    "varahamihira",
)

_HEADER_FUZZY_RE = re.compile(
    r"^\s*(brhat|brihat)\s+sa[mrnh]+[hi]?ita\b.*$",
    re.IGNORECASE,
)


def drop_devanagari(line: str) -> bool:
    """Return True if the line is majority Devanagari (U+0900-U+097F)."""
    if not line:
        return False
    devanagari_chars = sum(1 for c in line if "ऀ" <= c <= "ॿ")
    total_non_whitespace = sum(1 for c in line if not c.isspace())
    if total_non_whitespace == 0:
        return False
    # 'Majority' = strictly more than half the non-whitespace glyphs.
    return devanagari_chars * 2 > total_non_whitespace


def _is_page_number_only(line: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(line))


def _is_running_header(line: str) -> bool:
    lower = line.strip().lower()
    if not lower:
        return False
    if _HEADER_FUZZY_RE.match(line):
        return True
    if any(d in lower for d in _HEADER_DENYLIST) and len(lower) < 80:
        # A short line dominated by the book title is almost certainly a
        # header. Longer lines might be a citation in the body.
        return True
    return False


def _is_ocr_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    n_alpha = sum(1 for c in stripped if c.isalpha())
    return n_alpha < 3


def _dehyphenate(lines: Iterable[str]) -> list[str]:
    """Join lines where a word was split at a line break (``...end-`` + ``next``)."""
    out: list[str] = []
    buf: str | None = None
    for line in lines:
        if buf is not None:
            # Glue the previous truncated word to this line's first word.
            stripped = line.lstrip()
            if stripped:
                first, sep, rest = stripped.partition(" ")
                joined = buf + first
                if sep:
                    joined = joined + " " + rest
                out.append(joined)
            else:
                out.append(buf)
            buf = None
            continue
        if line.rstrip().endswith("-") and len(line.rstrip()) > 2:
            # Save the trailing-hyphen line's content (sans hyphen) for the next.
            buf = line.rstrip()[:-1]
        else:
            out.append(line)
    if buf is not None:
        out.append(buf)
    return out


def clean_text(raw: str) -> str:
    """Apply the full cleaning pipeline to one page's (or chapter's) OCR output.

    Steps:

    1. Split into lines.
    2. Drop Devanagari-majority lines, page-number-only lines, running
       headers/footers, and OCR-noise lines.
    3. De-hyphenate line-break-split words.
    4. Collapse multiple blank lines into one and trim each surviving
       line's internal whitespace runs.
    """
    if not raw:
        return ""

    lines = raw.splitlines()
    kept: list[str] = []
    for line in lines:
        if drop_devanagari(line):
            continue
        if _is_page_number_only(line):
            continue
        if _is_running_header(line):
            continue
        if _is_ocr_noise(line):
            continue
        kept.append(line)

    glued = _dehyphenate(kept)

    # Collapse internal whitespace runs; preserve paragraph breaks.
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in glued:
        squeezed = re.sub(r"\s+", " ", line).strip()
        if squeezed:
            cleaned_lines.append(squeezed)
            prev_blank = False
        elif not prev_blank:
            cleaned_lines.append("")
            prev_blank = True
    # Strip leading/trailing blanks.
    while cleaned_lines and not cleaned_lines[0]:
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


__all__ = [
    "clean_text",
    "drop_devanagari",
]
