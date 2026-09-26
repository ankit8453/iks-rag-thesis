"""Extract the advisory chapters of the NITI Aayog natural-farming manual into the
corpus as a `ready_external` book.

Source: newbooks/Training-Manual-English.pdf -- "Empowering Farmers: Natural Farming
Training Toolkit and Best Practices Guide", NITI Aayog, Feb 2026, ISBN 978-81-991080-0-4.
The PDF carries a clean text layer, so NO OCR (and no Gemini spend) is needed.

WHY this book. Phase 11 showed the limiter is corpus coverage, not retrieval: the
classical texts are thin on symptom -> remedy, so the generator honestly refuses ~55%
of answerable queries. This manual supplies exactly that missing layer, in structured
form (Purpose / Ingredients / Preparation / Application), and it is IKS-derived
(cow dung, cow urine, neem -- the kunapajala tradition) and government-published.

WHAT IS INGESTED (advisory content only):
    Ch 2  Seed Selection & Treatment          PDF pp. 36-43
    Ch 4  Soil Health Management              PDF pp. 60-65
    Ch 5  Pest and Disease Management         PDF pp. 66-77
    Ch 6  Bio-Input Production (6.1-6.2 only) PDF pp. 78-92
    Ch 8  Best Practices for 22 Key Crops     PDF pp. 112-165

WHAT IS SKIPPED, and why: Ch1 (NF theory), Ch3 (water conservation), Ch7
(certification), Ch9 (carbon credits), Ch10-11 (frameworks/schemes) -- none give
plant-level advice. Sections 6.3-6.4 (Bio-Input Resource Centre infrastructure and
government schemes, pp.93-103) are skipped for the same reason as Ch7/9/10/11.

CHUNKING NOTE. Each numbered section is emitted as ONE block, prefixed with its
chapter and section title, and containing no blank lines. The corpus chunker packs
whole paragraphs (200-500 tokens), so a formulation such as Neemastra stays intact
rather than being split away from its ingredients.

Usage:  python scripts/extract_niti_manual.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PDF = ROOT / "newbooks" / "Training-Manual-English.pdf"
OUT = ROOT / "corpus" / "ocr_external" / "niti_natural_farming.md"

# (first_page, last_page, chapter_no, chapter_title)  -- 1-based PDF pages
RANGES: list[tuple[int, int, int, str]] = [
    (36, 43, 2, "Seed Selection and Treatment"),
    (60, 65, 4, "Soil Health Management"),
    (66, 77, 5, "Pest and Disease Management"),
    (78, 92, 6, "Bio-Input Production Methods"),
    (112, 165, 8, "Best Practices for Key Crops"),
]

RUNNING_HEADER = "Empowering Farmers: Natural Farming Training Toolkit"
SECTION_RE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z][^\n]{2,70}?)\s*$")


def strip_footnotes(s: str) -> str:
    """Remove footnote URLs and their leading marker digits.

    Footnotes printed at the page foot extract inline, e.g. "a simple 20
    https://naturalfarming.dac.gov.in/... fermentation process", which corrupts the
    sentence and would pollute a retrieved chunk.
    """
    s = re.sub(r"\s*\d{0,3}\s*https?://\S+", " ", s)      # marker + URL
    s = re.sub(r"\s*www\.\S+", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def clean_page(text: str) -> str:
    """Drop the running header / bare page numbers, normalise justified tabs."""
    out: list[str] = []
    for line in text.replace("\t", " ").splitlines():
        s = line.strip()
        if not s:
            continue
        if RUNNING_HEADER.lower() in s.lower():
            continue
        if re.fullmatch(r"\d{1,3}", s):          # bare folio number
            continue
        if re.fullmatch(r"(Fig|Table)\s*\d+\.\d+.*", s, re.I):   # figure/table captions
            continue
        s = strip_footnotes(s)
        if not s or re.fullmatch(r"[\d\s.,;:-]+", s):
            continue
        out.append(re.sub(r"\s{2,}", " ", s))
    return "\n".join(out)


def tidy_title(title: str) -> str:
    """Strip footnote markers glued to a heading, e.g. 'Amla26' -> 'Amla'."""
    return re.sub(r"\s*\d{1,3}$", "", title).strip()


def join_body(lines: list[str]) -> str:
    """Join a section's lines into one blank-line-free block.

    Bullets keep their own line (readable, and still one paragraph to the chunker);
    ordinary prose lines are re-joined, repairing words hyphenated across a linebreak.
    """
    buf: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        is_bullet = s.startswith(("•", "-", "*")) or re.match(r"^\(?[ivx]+\)|^\(\d+\)", s)
        if is_bullet or not buf:
            buf.append(s)
            continue
        prev = buf[-1]
        if prev.startswith(("•", "-", "*")) and not is_bullet:
            buf[-1] = prev.rstrip("-") + (" " if not prev.endswith("-") else "") + s
        elif prev.endswith("-"):
            buf[-1] = prev[:-1] + s
        else:
            buf[-1] = prev + " " + s
    return "\n".join(buf)


def main() -> int:
    from pypdf import PdfReader

    if not PDF.is_file():
        raise SystemExit(f"PDF not found: {PDF}")
    reader = PdfReader(str(PDF))

    blocks: list[str] = []
    per_chapter: dict[int, int] = {}

    for first, last, chap_no, chap_title in RANGES:
        pages = [clean_page(reader.pages[p - 1].extract_text() or "")
                 for p in range(first, last + 1)]
        text = "\n".join(pages)

        # split the chapter text into its numbered sections
        cur_head: str | None = None
        cur_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        for line in text.splitlines():
            m = SECTION_RE.match(line)
            if m and m.group(1).split(".")[0] == str(chap_no):
                if cur_head is not None:
                    sections.append((cur_head, cur_lines))
                cur_head = f"{m.group(1)} {tidy_title(m.group(2))}"
                cur_lines = []
            elif cur_head is not None:
                cur_lines.append(line)
        if cur_head is not None:
            sections.append((cur_head, cur_lines))

        for head, lines in sections:
            body = join_body(lines)
            if len(body.split()) < 25:          # drop stubs / stray headings
                continue
            blocks.append(f"Chapter {chap_no}, {chap_title} - Section {head}.\n{body}")
            per_chapter[chap_no] = per_chapter.get(chap_no, 0) + 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    total_words = sum(len(b.split()) for b in blocks)
    print(f"sections written : {len(blocks)}")
    for c in sorted(per_chapter):
        print(f"   chapter {c}: {per_chapter[c]} sections")
    print(f"total words      : {total_words:,}")
    print(f"written          : {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
