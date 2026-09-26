"""Build a validation worksheet for the Kashyapiya Krishisukti translation.

WHY. Our Kashyapiya English text came from Gemini Flash 3.6 transcribing and
translating a Sanskrit-only scan (857 verses). Dr. Pandey then supplied the
Chowkhamba edition (Kashyapiya-Krishi-Paddhati, ed./tr. Dr. Shrikrishna 'Jugnu',
2013), which prints the same verses with a Hindi commentary. That edition is the
independent check on our AI translation -- and answers "what is the benefit of
adding this book" with evidence rather than assertion.

WHAT THIS PRODUCES. A worksheet sampling agriculturally meaningful verses spread
across the text. For each verse it shows the Sanskrit we OCR'd (so the verse can be
located in the Chowkhamba edition by eye) and our English rendering, with a blank
verdict column. Ankit reads the Hindi commentary and marks agree / partly / wrong.

Verse -> page hint: verses 39-51 sit on Chowkhamba PDF pp.60-62, i.e. roughly 4-5
verses per page from p.60, so page ~= 60 + (verse - 39) / 4.5. This is an estimate
to start the search, not an index.

Usage:  python scripts/build_kashyapiya_validation.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "sanskrit_ocr_demo" / "pages"
OUT_MD = ROOT / "seminar" / "kashyapiya_translation_validation.md"
OUT_DOCX = ROOT / "seminar" / "kashyapiya_translation_validation.docx"

# agricultural themes worth validating (the content the thesis actually uses)
THEMES = [
    ("soil", ["soil", "land", "earth", "bhumi"]),
    ("water", ["water", "rain", "well", "canal", "reservoir", "irrigat"]),
    ("seed", ["seed", "sow", "sprout", "germinat"]),
    ("crop", ["crop", "grain", "rice", "paddy", "sasya", "harvest"]),
    ("tree/plant", ["tree", "plant", "leaf", "root", "fruit", "flower"]),
    ("pest/disease", ["pest", "insect", "disease", "worm", "damage", "protect"]),
]


def load_verses() -> list[dict]:
    """Parse [verse N] / SA: / EN: blocks from every OCR'd page."""
    out: list[dict] = []
    for f in sorted(PAGES.glob("page_*.txt")):
        page = int(f.stem.split("_")[1])
        cur: dict | None = None
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            m = re.match(r"^\[verse\s+([0-9]+)", s, re.I)
            if m:
                if cur and cur.get("en"):
                    out.append(cur)
                cur = {"verse": int(m.group(1)), "page": page, "sa": "", "en": ""}
            elif cur is not None and s.startswith("SA:"):
                cur["sa"] = s[3:].strip()
            elif cur is not None and s.startswith("EN:"):
                cur["en"] = s[3:].strip()
        if cur and cur.get("en"):
            out.append(cur)
    return out


def hindi_page(our_page: int) -> int:
    """Estimated page in the Chowkhamba edition, from OUR scan page.

    NOTE: verse numbers RESTART per section in this text (a "verse 42" occurs on
    our page 4 and again on page 48), so the verse number alone cannot locate a
    passage -- the estimate must come from position in the scan, not the number.

    Anchor: our scan p.4 (verses 28-42) corresponds to Chowkhamba pp.60-62. Our 64
    scanned pages span roughly 165 Chowkhamba content pages, i.e. ~2.6 of theirs per
    one of ours.
    """
    est = 60 + (our_page - 4) * 2.6
    return int(max(57, min(225, round(est))))


def pick_samples(verses: list[dict], per_theme: int = 2) -> list[tuple[str, dict]]:
    """Pick verses per theme, forced to spread across the whole text.

    The opening verses are a traditional preamble (kings, dharma) and are poor
    validation material, so candidates are bucketed by position in the text and one
    is taken per bucket -- guaranteeing early, middle and late coverage.
    """
    lo = min(v["verse"] for v in verses)
    hi = max(v["verse"] for v in verses)
    span = max(1, hi - lo)
    skip_preamble = lo + 0.03 * span          # drop the invocation/preamble

    picked: list[tuple[str, dict]] = []
    used: set[int] = set()
    for theme, kws in THEMES:
        hits = [v for v in verses
                if v["verse"] not in used and v["verse"] >= skip_preamble
                and any(k in v["en"].lower() for k in kws)
                and 20 <= len(v["en"].split()) <= 60 and v["sa"]]
        if not hits:
            continue
        hits.sort(key=lambda v: v["verse"])
        # one hit per bucket, buckets spanning the text
        for b in range(per_theme):
            b_lo = skip_preamble + (hi - skip_preamble) * b / per_theme
            b_hi = skip_preamble + (hi - skip_preamble) * (b + 1) / per_theme
            in_bucket = [v for v in hits
                         if b_lo <= v["verse"] < b_hi and v["verse"] not in used]
            if in_bucket:
                v = in_bucket[len(in_bucket) // 2]     # middle of the bucket
                picked.append((theme, v))
                used.add(v["verse"])
    # sort by position in the scan, NOT by verse number (numbers restart per section)
    picked.sort(key=lambda t: (t[1]["page"], t[1]["verse"]))
    return picked


def main() -> int:
    verses = load_verses()
    if not verses:
        raise SystemExit("no verses found - run the Kashyapiya OCR first")
    samples = pick_samples(verses)

    # ---------- markdown ----------
    md = [
        "# Kashyapiya Krishisukti - translation validation worksheet\n",
        f"\nOur English text was produced by Gemini Flash 3.6 from a Sanskrit-only scan "
        f"({len(verses)} verses OCR'd). This worksheet checks it against the **Chowkhamba "
        "edition** (*Kashyapiya-Krishi-Paddhati*, ed./tr. Dr. Shrikrishna 'Jugnu', 2013, "
        "`newbooks/Kashyapiya-Krishi-Paddhati.pdf`), which prints the same verses with a "
        "Hindi commentary.\n",
        "\n**How to use.** For each verse below, find it in the Chowkhamba PDF, read the "
        "Hindi commentary, and mark the verdict: **agree / partly / wrong**. Note anything "
        "the Hindi says that our English missed.\n"
        "\n> **Important:** verse numbers **restart in each section** of this text (a "
        "\"verse 42\" occurs in more than one place), so the number alone will not find the "
        "passage - **match the Sanskrit line by eye**. The page column is a starting "
        "estimate only, derived from position in our scan.\n",
        f"\n| # | Verse | Theme | Chowkhamba p. (est.) | Verdict | Notes |\n"
        "|---|---|---|---|---|---|\n",
    ]
    for i, (theme, v) in enumerate(samples, 1):
        md.append(f"| {i} | {v['verse']} | {theme} | ~{hindi_page(v['page'])} |  |  |\n")

    md.append("\n---\n\n## The verses\n")
    for i, (theme, v) in enumerate(samples, 1):
        md.append(
            f"\n### {i}. Verse {v['verse']}  ({theme})\n\n"
            f"- **Chowkhamba page (estimate):** ~{hindi_page(v['page'])}  \n"
            f"- **Our OCR page:** `sanskrit_ocr_demo/pages/page_{v['page']:04d}.txt`\n\n"
            f"**Sanskrit (our OCR):**\n\n> {v['sa']}\n\n"
            f"**Our English (Gemini):**\n\n> {v['en']}\n\n"
            f"**Verdict:** ______  **Notes:** ____________________________________\n"
        )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md), encoding="utf-8")

    # ---------- docx ----------
    try:
        from docx import Document
        from docx.shared import Pt

        doc = Document()
        st = doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        doc.add_heading("Kashyapiya Krishisukti - translation validation", level=1)
        doc.add_paragraph(
            f"Our English came from Gemini Flash 3.6 on a Sanskrit-only scan ({len(verses)} "
            "verses). Checked against the Chowkhamba edition (Dr. Shrikrishna 'Jugnu', 2013), "
            "which prints the same verses with a Hindi commentary."
        )
        doc.add_paragraph(
            "For each verse: find it in the Chowkhamba PDF (page hint gets you close; match "
            "the Sanskrit by eye), read the Hindi, mark agree / partly / wrong."
        )
        t = doc.add_table(rows=1, cols=6)
        t.style = "Table Grid"
        for c, h in zip(t.rows[0].cells,
                        ["#", "Verse", "Theme", "Chowkhamba p.", "Verdict", "Notes"]):
            c.text = h
        for i, (theme, v) in enumerate(samples, 1):
            r = t.add_row().cells
            r[0].text, r[1].text, r[2].text = str(i), str(v["verse"]), theme
            r[3].text, r[4].text, r[5].text = f"~{hindi_page(v['page'])}", "", ""
        doc.add_page_break()
        doc.add_heading("The verses", level=2)
        for i, (theme, v) in enumerate(samples, 1):
            doc.add_heading(f"{i}. Verse {v['verse']} ({theme})", level=3)
            doc.add_paragraph(f"Chowkhamba page (estimate): ~{hindi_page(v['page'])}")
            p = doc.add_paragraph(); p.add_run("Sanskrit (our OCR): ").bold = True
            p.add_run(v["sa"])
            p = doc.add_paragraph(); p.add_run("Our English (Gemini): ").bold = True
            p.add_run(v["en"])
            doc.add_paragraph("Verdict: ____________   Notes: ____________________________")
        doc.save(OUT_DOCX)
        made_docx = True
    except Exception as exc:  # noqa: BLE001
        made_docx = False
        print("  ! docx not written:", exc)

    print(f"verses parsed  : {len(verses)}")
    print(f"samples chosen : {len(samples)}  (verses {[v['verse'] for _, v in samples]})")
    print(f"written        : {OUT_MD.relative_to(ROOT)}")
    if made_docx:
        print(f"                 {OUT_DOCX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
