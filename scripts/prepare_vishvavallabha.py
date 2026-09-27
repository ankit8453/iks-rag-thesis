"""Turn the archive.org OCR of Vishvavallabha into a clean, chapter-marked text (§6q).

Source: ``newbooks/vishvavallabha/Vishvavallabha_archive_ocr.txt`` — the Internet Archive's
Tesseract OCR of the AAHF 2004 edition (tr. Nalini Sadhale). No Gemini call, no cost.

What is kept. Only the **English translation**: from "Chapter 1: Groundwater" up to the
start of "Commentaries". The Sanskrit pages are not used — Tesseract read the Devanagari as
Cyrillic, so they are unusable as OCR — and the commentaries are the translators' notes,
not the treatise.

Why chapter markers. Vishvavallabha restarts its verse numbering in every chapter, and in
each of chapter I's five sub-parts (A–E). Chunked as one block, verse 9 of chapter I and
verse 9 of chapter VIII would carry the same citation. So each chapter is written under a
``## Chapter <label>: <title>`` line and ``build_corpus`` chunks them separately, giving
citations like ``[Vishvavallabha, ch.8, v.9-12]``. Chapter I's parts become 1A–1E, as in
the book's own verse table.

Cleaning, all measured before being applied (EXPERIMENT_LOG.md 6q):
- Cyrillic look-alikes that Tesseract's multi-language model dropped into Latin text
  (47 of 2,987 lines in this zone, mostly headings such as "Chapter У") are mapped back;
- stray OCR marks ("<", ">", "SS") and bare page numbers are dropped;
- colophons ("Thus ends Ullasa V of Vishvavallabha composed by Shri Mishra Chakrapani")
  are dropped — they carry no content and would otherwise merge into the last verse;
- editorial notes such as "[*Defective text.]" are KEPT: they tell a reader where the
  manuscript itself is damaged.

Output: ``corpus/ocr_external/vishvavallabha.md`` (gitignored — copyrighted translation).

Usage:  python scripts/prepare_vishvavallabha.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "newbooks" / "vishvavallabha" / "Vishvavallabha_archive_ocr.txt"
OUT = ROOT / "corpus" / "ocr_external" / "vishvavallabha.md"

#: Chapter titles from the book's own table of contents.
TITLES = {
    "1": "Groundwater", "2": "Water reservoirs",
    "3": "Examination and suitability of ground", "4": "Propagation and plantation",
    "5": "Water management", "6": "Protection and care", "7": "Nourishment and growth",
    "8": "Diseases and treatment", "9": "Botanical wonders",
}
ROMAN = {"I": "1", "Il": "2", "II": "2", "III": "3", "IV": "4", "V": "5",
         "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "1": "1"}

#: Cyrillic characters that are visual twins of Latin ones in this OCR.
CYR = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "V", "Ш": "III", "П": "II",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    # italic-glyph twins found in body text on the first pass: "aп"->an, "oг"->or,
    # "aтгa"->amra (mango), "гoo!"->roo(t), "Бaдaтa"->badama, "oйcaкe"->oi(l)cake
    "п": "n", "г": "r", "т": "m", "д": "d", "Б": "B", "к": "k", "з": "s", "л": "l",
    "й": "i", "Г": "F",
})

_CHAPTER = re.compile(r"^Chapter\s+([IVXl1]+)\s*:\s*(.*)$")
_PART = re.compile(r"^([A-E])\.\s+[A-Z]")
_VERSE = re.compile(r"^\d{1,3}\.\s+\S")
_NOISE = re.compile(r"^(?:[<>]+|SS|\d{1,4})$")
_COLOPHON = re.compile(r"^(?:Thus ends\b.*|(?:by\s+)?Shri Mishra Chakrapani\.?)$")


def main() -> int:
    if not SRC.is_file():
        print(f"missing source: {SRC.relative_to(ROOT)}")
        return 2
    lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("Chapter 1: Groundwater"))
    end = next(i for i, l in enumerate(lines)
               if l.strip().startswith("Commentaries") and i > start)

    out: list[str] = []
    chapter = None
    skipping_title_tail = False
    stats: dict[str, int] = {}

    for raw in lines[start:end]:
        line = raw.translate(CYR).strip()
        m = _CHAPTER.match(line)
        if m:
            chapter = ROMAN.get(m.group(1), m.group(1))
            label = "1A" if chapter == "1" else chapter
            title = TITLES[chapter] + (", part A" if chapter == "1" else "")
            out += ["", f"## Chapter {label}: {title}", ""]
            skipping_title_tail = True        # drop a wrapped title's second line
            stats[label] = 0
            continue
        if chapter == "1":
            pm = _PART.match(line)
            if pm and pm.group(1) != "A":
                label = f"1{pm.group(1)}"
                out += ["", f"## Chapter {label}: Groundwater, part {pm.group(1)}", ""]
                skipping_title_tail = True
                stats[label] = 0
                continue
            if pm:                              # "A. Groundwater" heading of part A
                continue
        if not line:
            out.append("")
            continue
        if _NOISE.match(line) or _COLOPHON.match(line):
            continue
        if skipping_title_tail:
            # between a heading and its first verse, short lines are title fragments or
            # sub-headings ("treatments", "Protection"); a verse ends the skip
            if _VERSE.match(line):
                skipping_title_tail = False
            elif len(line.split()) <= 6:
                continue
            else:
                skipping_title_tail = False
        if _VERSE.match(line) and stats:
            stats[list(stats)[-1]] += 1
        out.append(line)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    header = ("<!-- Vishvavallabha (Chakrapani Mishra, c.1577), tr. Nalini Sadhale, "
              "AAHF Agri-History Bulletin No. 5, 2004. English translation only, from the "
              "Internet Archive OCR; prepared by scripts/prepare_vishvavallabha.py. "
              "Copyrighted - do not commit. -->\n\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + text, encoding="utf-8")

    residue = sum(1 for l in text.splitlines() if re.search(r"[Ѐ-ӿ]", l))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(text.split()):,} words)")
    print(f"{'chapter':8} verses")
    for k, v in stats.items():
        print(f"  {k:6} {v:4}")
    print(f"total verse starts: {sum(stats.values())}")
    print(f"lines still containing Cyrillic: {residue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
