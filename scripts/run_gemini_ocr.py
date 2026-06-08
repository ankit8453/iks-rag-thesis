"""Phase 3b.2 — Gemini 3.5 Flash re-OCR for all 4 IKS corpus books.

Replaces Tesseract OCR for Vrikshayurveda + Brihat Samhita and supplies
the cleaned-text source for the two new books (Krishi Parashara +
Upavanavinoda) declared as ``ready_external`` in books.yaml.

Outputs are written to the exact paths the existing pipeline expects:

- ``vrikshayurveda`` -> ``corpus/raw/vrikshayurveda/page_NNNN.txt`` (one
  file per page; build_corpus reads these as the Tesseract cache once
  Tesseract is skipped because the file is already present and non-stale).
- ``brihat_samhita`` -> ``corpus/raw/brihat_samhita/page_NNNN.txt`` for
  PDF pages 275..593 (the union span of the 12 wanted chapters). Pages
  1..274 are written as empty .txt files so build_corpus' OCR cache
  hits and no Tesseract is triggered.
- ``krishi_parashara`` -> ``corpus/ocr_external/krishi_parashara.md``
  (single Markdown file, per the Phase 3b external-OCR contract).
- ``upavanavinoda`` -> ``corpus/ocr_external/upavanavinoda.md`` (same).

Idempotency: every per-page Gemini output is written immediately, so a
crash/restart resumes from the next un-processed page with zero
re-spend. The merged .md files for the two external books are
assembled at the very end from the same per-page cache.

Cost tracking: every Gemini response's ``usage_metadata`` is parsed,
input/output tokens summed, and a running rupee tally printed every 25
pages. A configurable hard ceiling stops the run early if the actual
rate trends above budget.

Usage: ``python scripts/run_gemini_ocr.py`` from the repo root.

Environment: requires ``GEMINI_API_KEY`` in ``.env.gemini`` at the repo
root (gitignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.rag.corpus.ocr import _resolve_poppler_path
from src.utils.logging_setup import get_logger
from src.utils.paths import CONFIGS_DIR, CORPUS_RAW_DIR, PROJECT_ROOT

_LOGGER = get_logger(__name__)

# ----------------------------- pricing ---------------------------------
#
# Gemini 3.5 Flash paid tier (USD per 1M tokens, 2026 rates). If
# pricing changes update HERE; the rupee math below stays the same.
PRICE_INPUT_PER_M_USD: float = 0.30      # text + image input
PRICE_OUTPUT_PER_M_USD: float = 2.50     # generated output
USD_TO_INR: float = 84.0                  # rough mid-2026
HARD_BUDGET_INR: float = 200.0            # safety stop

# ----------------------------- prompt ---------------------------------

OCR_PROMPT: str = (
    "You are transcribing one page of a scanned printed English "
    "translation of a classical Sanskrit treatise (Vrikshayurveda, "
    "Brihat Samhita, Krishi Parashara, or Upavanavinoda).\n"
    "\n"
    "Output ONLY the English translation text on this page. Rules:\n"
    "1. Skip Devanagari script entirely. Do not transliterate it. If a "
    "verse appears in Devanagari followed by the English translation, "
    "output only the English translation.\n"
    "2. Skip running headers and footers. Skip standalone page numbers.\n"
    "3. Preserve verse / section numbers exactly as printed (e.g. '1.', "
    "'24.', 'XII.', '(3)'). Each numbered verse should start on a new "
    "line.\n"
    "4. Preserve paragraph breaks. Output plain text, NOT Markdown.\n"
    "5. For Sanskrit plant or term names (e.g. 'Asvattha', 'Bilva', "
    "'Kunapajala'), use the spelling shown on this page; if the scan "
    "is smudged or ambiguous, use the spelling most consistent with "
    "surrounding context. Do NOT invent content not present on the "
    "page.\n"
    "6. If the page is blank, contains only Devanagari, or is pure "
    "front-matter / publisher info / index / glossary / footnotes with "
    "no translated body text, output a single line: [SKIP]\n"
)

# ----------------------------- jobs ------------------------------------


@dataclass
class OCRJob:
    """One book's Gemini re-OCR spec."""

    book_id: str
    pdf_rel: str
    pages_to_ocr: list[int]              # 1-based PDF pages
    output_kind: str                      # "per_page" or "merged_md"
    per_page_cache_dir: Path              # where each Gemini page goes
    merged_md_path: Path | None = None    # for output_kind == "merged_md"
    empty_pad_pages: list[int] = field(default_factory=list)  # only for per_page books
    empty_pad_dir: Path | None = None


def _brihat_union_span(books_cfg: dict) -> tuple[int, int]:
    """Return (first, last) PDF-page range covering all wanted Brihat
    chapters. Uses the previously-computed Tesseract spans 275..593.

    Hard-coded here to avoid re-running locate_chapters (which would
    require the Tesseract cache that we're about to delete). The 12
    wanted chapters span PDF pages 275..593; this was verified in the
    Phase 3b prep step.
    """
    # Sanity: assert the config still asks for the same 12 chapters.
    brihat = next(b for b in books_cfg["books"] if b["id"] == "brihat_samhita")
    wanted = sorted(brihat.get("chapters") or [])
    assert wanted == [21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 54, 55], (
        f"Brihat chapter list changed: {wanted}. Re-derive the union span "
        "before running Gemini OCR."
    )
    return 275, 593


def _build_jobs() -> list[OCRJob]:
    books_cfg = yaml.safe_load(
        (CONFIGS_DIR / "corpus" / "books.yaml").read_text(encoding="utf-8")
    )
    books = {b["id"]: b for b in books_cfg["books"]}

    jobs: list[OCRJob] = []

    # ---- Vrikshayurveda: scope=full, all 101 pages -----------------
    vrik = books["vrikshayurveda"]
    jobs.append(OCRJob(
        book_id="vrikshayurveda",
        pdf_rel=vrik["pdf"],
        pages_to_ocr=list(range(1, 102)),
        output_kind="per_page",
        per_page_cache_dir=CORPUS_RAW_DIR / "vrikshayurveda",
    ))

    # ---- Brihat Samhita: scope=chapters, union span 275..593 -------
    brihat_first, brihat_last = _brihat_union_span(books_cfg)
    brihat_pages = list(range(brihat_first, brihat_last + 1))
    brihat_pad_pages = list(range(1, brihat_first))  # 1..274 empty
    jobs.append(OCRJob(
        book_id="brihat_samhita",
        pdf_rel=books["brihat_samhita"]["pdf"],
        pages_to_ocr=brihat_pages,
        output_kind="per_page",
        per_page_cache_dir=CORPUS_RAW_DIR / "brihat_samhita",
        empty_pad_pages=brihat_pad_pages,
        empty_pad_dir=CORPUS_RAW_DIR / "brihat_samhita",
    ))

    # ---- Krishi Parashara: scope=pages, 94..119 --------------------
    kp = books["krishi_parashara"]
    pr_kp = kp["page_range"]
    jobs.append(OCRJob(
        book_id="krishi_parashara",
        pdf_rel=kp["pdf"],
        pages_to_ocr=list(range(pr_kp[0], pr_kp[1] + 1)),
        output_kind="merged_md",
        per_page_cache_dir=CORPUS_RAW_DIR / "krishi_parashara",
        merged_md_path=PROJECT_ROOT / kp["text_source"],
    ))

    # ---- Upavanavinoda: scope=pages, 77..96 ------------------------
    uv = books["upavanavinoda"]
    pr_uv = uv["page_range"]
    jobs.append(OCRJob(
        book_id="upavanavinoda",
        pdf_rel=uv["pdf"],
        pages_to_ocr=list(range(pr_uv[0], pr_uv[1] + 1)),
        output_kind="merged_md",
        per_page_cache_dir=CORPUS_RAW_DIR / "upavanavinoda",
        merged_md_path=PROJECT_ROOT / uv["text_source"],
    ))

    return jobs


# ----------------------------- gemini ---------------------------------


def _load_api_key() -> str:
    env_file = PROJECT_ROOT / ".env.gemini"
    if not env_file.is_file():
        raise FileNotFoundError(f"{env_file} not found")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'").strip('"')
    raise ValueError("GEMINI_API_KEY not found in .env.gemini")


def _page_cache_path(job: OCRJob, page_number: int) -> Path:
    return job.per_page_cache_dir / f"page_{page_number:04d}.txt"


def _is_already_done(job: OCRJob, page_number: int) -> bool:
    """A page is 'done' if its cache file exists. Empty file counts as
    a deliberate [SKIP] result (saved as empty during a prior run)."""
    p = _page_cache_path(job, page_number)
    return p.is_file()


def _ocr_one_page(
    client, model_name: str, pdf_path: Path, page_number: int, poppler: str | None,
) -> tuple[str, dict]:
    """Run Gemini OCR on one PDF page. Returns (text, usage_dict)."""
    from pdf2image import convert_from_path  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415

    kwargs = {"dpi": 220, "first_page": page_number, "last_page": page_number}
    if poppler:
        kwargs["poppler_path"] = poppler
    images = convert_from_path(str(pdf_path), **kwargs)
    if not images:
        return "", {"input_tokens": 0, "output_tokens": 0}

    resp = client.models.generate_content(
        model=model_name,
        contents=[OCR_PROMPT, images[0]],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    text = (resp.text or "").strip()

    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    else:
        in_tok, out_tok = 0, 0

    return text, {"input_tokens": in_tok, "output_tokens": out_tok}


def _cost_inr(in_tok: int, out_tok: int) -> float:
    usd = (in_tok / 1_000_000) * PRICE_INPUT_PER_M_USD + (out_tok / 1_000_000) * PRICE_OUTPUT_PER_M_USD
    return usd * USD_TO_INR


def _write_page_cache(job: OCRJob, page_number: int, text: str) -> None:
    job.per_page_cache_dir.mkdir(parents=True, exist_ok=True)
    payload = "" if text.strip() == "[SKIP]" else text
    _page_cache_path(job, page_number).write_text(payload, encoding="utf-8")


def _assemble_merged_md(job: OCRJob) -> int:
    """For merged_md jobs, concatenate all per-page caches into a single
    .md file (skipping empty / [SKIP] pages). Returns total chars
    written."""
    parts: list[str] = []
    for pg in job.pages_to_ocr:
        path = _page_cache_path(job, pg)
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8").strip()
        if not body:
            continue
        parts.append(body)
    merged = "\n\n".join(parts).strip() + "\n"
    assert job.merged_md_path is not None
    job.merged_md_path.parent.mkdir(parents=True, exist_ok=True)
    job.merged_md_path.write_text(merged, encoding="utf-8")
    return len(merged)


def _write_empty_pad_pages(job: OCRJob) -> int:
    """For Brihat: write empty .txt files for pages outside the wanted
    span so build_corpus' OCR cache hits and Tesseract is never run."""
    if not job.empty_pad_pages or job.empty_pad_dir is None:
        return 0
    job.empty_pad_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for pg in job.empty_pad_pages:
        p = job.empty_pad_dir / f"page_{pg:04d}.txt"
        if not p.is_file():
            p.write_text("", encoding="utf-8")
            written += 1
    return written


# ----------------------------- main loop -------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default="gemini-3.5-flash",
        help="Gemini model name (default: gemini-3.5-flash)",
    )
    parser.add_argument(
        "--throttle-seconds", type=float, default=1.5,
        help="Seconds to sleep between API calls (default: 1.5)",
    )
    parser.add_argument(
        "--dry-run-one-each", action="store_true",
        help="OCR only the first page of each book (sanity test).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated book ids to run (default: all 4).",
    )
    args = parser.parse_args(argv)

    from google import genai  # noqa: PLC0415

    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    poppler = _resolve_poppler_path()
    _LOGGER.info("Poppler: %s", poppler or "(on PATH)")
    _LOGGER.info("Model: %s | throttle: %.2fs", args.model, args.throttle_seconds)

    jobs = _build_jobs()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        jobs = [j for j in jobs if j.book_id in wanted]
    if args.dry_run_one_each:
        for j in jobs:
            j.pages_to_ocr = j.pages_to_ocr[:1]
            j.empty_pad_pages = []   # don't pad on dry run

    total_pages = sum(len(j.pages_to_ocr) for j in jobs)
    _LOGGER.info("Total Gemini pages to OCR: %d across %d book(s)",
                 total_pages, len(jobs))

    # Logging setup
    file_handler = logging.FileHandler(PROJECT_ROOT / "scripts" / "_gemini_ocr.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)

    total_in_tok = 0
    total_out_tok = 0
    pages_done_this_run = 0
    pages_cached = 0
    t0 = time.monotonic()

    for job in jobs:
        pdf_path = PROJECT_ROOT / job.pdf_rel
        if not pdf_path.is_file():
            _LOGGER.error("PDF missing for %s: %s", job.book_id, pdf_path)
            return 2
        job.per_page_cache_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("=" * 70)
        _LOGGER.info("Book: %s (%d pages, %s)", job.book_id,
                     len(job.pages_to_ocr), job.output_kind)

        for idx, pg in enumerate(job.pages_to_ocr, 1):
            if _is_already_done(job, pg):
                pages_cached += 1
                continue

            try:
                text, usage = _ocr_one_page(
                    client, args.model, pdf_path, pg, poppler,
                )
            except Exception as exc:
                _LOGGER.error(
                    "%s p.%d: Gemini call failed: %s", job.book_id, pg, exc,
                )
                # one retry after a 5s backoff
                time.sleep(5.0)
                try:
                    text, usage = _ocr_one_page(
                        client, args.model, pdf_path, pg, poppler,
                    )
                except Exception as exc2:
                    _LOGGER.error(
                        "%s p.%d: retry failed too, skipping: %s",
                        job.book_id, pg, exc2,
                    )
                    continue

            _write_page_cache(job, pg, text)
            total_in_tok += usage["input_tokens"]
            total_out_tok += usage["output_tokens"]
            pages_done_this_run += 1

            if pages_done_this_run % 25 == 0 or idx == len(job.pages_to_ocr):
                spent_inr = _cost_inr(total_in_tok, total_out_tok)
                elapsed = time.monotonic() - t0
                _LOGGER.info(
                    "  progress: %s p.%d (%d/%d) | "
                    "this-run done=%d cached=%d | "
                    "tokens in=%d out=%d | spent=%.2f INR | t=%.0fs",
                    job.book_id, pg, idx, len(job.pages_to_ocr),
                    pages_done_this_run, pages_cached,
                    total_in_tok, total_out_tok, spent_inr, elapsed,
                )
                if spent_inr > HARD_BUDGET_INR:
                    _LOGGER.error(
                        "Hard budget %.0f INR exceeded (spent %.2f). "
                        "Stopping. Resume by re-running this script.",
                        HARD_BUDGET_INR, spent_inr,
                    )
                    return 3

            time.sleep(args.throttle_seconds)

        # Assemble merged_md or pad empty pages
        if job.output_kind == "merged_md":
            n_chars = _assemble_merged_md(job)
            _LOGGER.info(
                "  assembled %s -> %s (%d chars)",
                job.book_id, job.merged_md_path, n_chars,
            )
        elif job.empty_pad_pages:
            n_padded = _write_empty_pad_pages(job)
            _LOGGER.info(
                "  padded %d empty .txt files outside wanted span for %s",
                n_padded, job.book_id,
            )

    spent_inr = _cost_inr(total_in_tok, total_out_tok)
    elapsed = time.monotonic() - t0
    print()
    print("=" * 70)
    print(f"Gemini OCR complete in {elapsed:.0f}s")
    print(f"  pages OCR'd this run: {pages_done_this_run}")
    print(f"  pages served from cache: {pages_cached}")
    print(f"  total input tokens : {total_in_tok:,}")
    print(f"  total output tokens: {total_out_tok:,}")
    print(f"  estimated spend    : {spent_inr:.2f} INR  "
          f"(${spent_inr / USD_TO_INR:.3f} USD)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
