# `corpus/ocr_external/` — externally-OCR'd cleaned text

This directory holds the cleaned English-translation text for books
whose OCR is done **outside** the Phase 3 Tesseract pipeline (typically
via Gemini Document AI). Each book has a single Markdown file named
`<book_id>.md` whose contents are read directly by
`src/rag/corpus/build_corpus.py` whenever the corresponding entry in
`configs/corpus/books.yaml` has:

```yaml
status: ready_external
ocr_method: gemini_external
text_source: corpus/ocr_external/<book_id>.md
```

## What goes in here

One Markdown file per book, with the cleaned English-translation text
for the PDF page range declared in `books.yaml` under `page_range`. For
example, `krishi_parashara.md` covers the English-translation block at
PDF pages 94–119 of the Majumdar 1960 edition.

The text should already be:

- English only (Devanagari skipped — Gemini handles this via prompt)
- Free of running headers / footers / standalone page numbers
- Verse-numbered (`1.`, `24.`, etc.) where the source preserves it, so
  the chunker picks up verse boundaries naturally

Whitespace formatting is light Markdown — paragraph breaks preserved,
no headings needed (the chunker derives `chapter` from the
`page_range` declaration in `books.yaml`).

## Currently expected files

| File | Source PDF | PDF pages | Status |
|---|---|---|---|
| `krishi_parashara.md` | `KrishiParasara-...-Majumdar...1960bis.pdf` | 94–119 | awaiting Gemini OCR |
| `upavanavinoda.md` | `2015.282467.Upavana-Vinoda.pdf` | 77–96 | awaiting Gemini OCR |

When a file is present here, the next `python -m src.rag.corpus.build_corpus`
run picks it up, chunks it, embeds it, and adds it to ChromaDB
(`iks_corpus` collection). When it's missing, the build logs an
"awaiting Gemini OCR at ..." line and skips the book without raising.

## Why this directory is mostly gitignored

The cleaned text is copyrighted translation content (Majumdar / AAHF /
MLBD editions) — master plan §38 forbids redistributing it via the
public GitHub repo. So `*.md` here is **gitignored**, only this
`README.md` is tracked. Chunks land in HF Hub via the private
`ankit-iiitdmj/iks-corpus-chunks` dataset (same §38-compliant
transport used for Vrik + Brihat).

## How to (re)generate one

Detailed Gemini OCR procedure lives in `scripts/reocr_with_gemini.py`
(or its successor). At a high level: render the relevant PDF page
range at 300 dpi, send each page image to `gemini-3.5-flash` with the
"transcribe English only, skip Devanagari, preserve verse numbers"
prompt, concatenate the per-page outputs in order, write to
`<book_id>.md`.
