"""Re-OCR ONE corpus book with Gemini Flash 3.6, no output-token cap.

Why this exists
---------------
The original 4-book OCR (scripts/run_gemini_ocr.py) used Gemini 3.5 with an
implicit output limit. On dense pages that truncated output, and at least one
page saved an API error string as if it were text
(corpus/raw/brihat_samhita/page_0332.txt = "An error occurred..."). This script
re-runs the SAME task (output the English translation only; skip Devanagari-only
and front/back matter) but with:

  * model = gemini-3.6-flash
  * NO max_output_tokens (nothing can truncate)
  * the saved-error string treated as a FAILURE (retried, never cached)

SAFETY: writes to a SEPARATE folder ``corpus/raw_reocr/<book>/`` — the live
corpus (``corpus/raw/<book>/``) is left untouched until you compare and approve.
Per-page cache => resumable, zero re-spend on restart.

Run ONE book at a time, verify, then the next.

Usage:
    python scripts/reocr_one_book.py vrikshayurveda
    python scripts/reocr_one_book.py vrikshayurveda --pages 41-93   # a subrange
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.rag.corpus.ocr import _resolve_poppler_path

# same prompt intent as the original run (English-only; skip Devanagari pages)
PROMPT = (
    "You are transcribing one page of a scanned printed English translation of a "
    "classical Sanskrit treatise (Vrikshayurveda, Brihat Samhita, Krishi "
    "Parashara, or Upavanavinoda).\n"
    "\n"
    "Output ONLY the English translation text on this page. Rules:\n"
    "1. Skip Devanagari script entirely. Do not transliterate it. If a verse "
    "appears in Devanagari followed by the English translation, output only the "
    "English translation.\n"
    "2. Skip running headers and footers. Skip standalone page numbers.\n"
    "3. Preserve verse / section numbers exactly as printed (e.g. '1.', '24.', "
    "'XII.', '(3)'). Each numbered verse should start on a new line.\n"
    "4. Preserve paragraph breaks. Output plain text, NOT Markdown.\n"
    "5. Transcribe the WHOLE page from top to bottom. Do not stop early, do not "
    "summarise, do not omit any translated line. A page may be long.\n"
    "6. KEEP the page whenever it has ANY readable English sentences or prose — "
    "this INCLUDES the Introduction, Preface, editor's notes, chapter "
    "descriptions, commentary, and explanatory paragraphs. These are valuable "
    "content, NOT skippable front-matter. When in doubt, transcribe it.\n"
    "7. Output the single line [SKIP] ONLY when the page has essentially NO "
    "English body text — i.e. it is genuinely blank, or shows only Devanagari "
    "script, or is nothing but a title page / a plain list of contents / a bare "
    "alphabetical index / a page of only numbers. Never skip a page that has "
    "real English sentences on it.\n"
)

# an API/refusal string that must NEVER be cached as if it were page text
_ERROR_MARKERS = (
    "an error occurred", "please try again", "i cannot", "i'm sorry",
    "as an ai", "unable to process", "cannot fulfil", "cannot fulfill",
)

PRICE_INPUT_PER_M_USD = 0.30
PRICE_OUTPUT_PER_M_USD = 2.50
USD_TO_INR = 84.0
HARD_BUDGET_INR = 300.0


def load_api_key() -> str:
    for line in (ROOT / ".env.gemini").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'").strip('"')
    raise ValueError("GEMINI_API_KEY not found in .env.gemini")


def cost_inr(i, o):
    return (i / 1e6 * PRICE_INPUT_PER_M_USD + o / 1e6 * PRICE_OUTPUT_PER_M_USD) * USD_TO_INR


def book_pdf_and_pages(book_id: str):
    cfg = yaml.safe_load((ROOT / "configs" / "corpus" / "books.yaml").read_text(encoding="utf-8"))
    b = next((x for x in cfg["books"] if x["id"] == book_id), None)
    if not b:
        raise SystemExit(f"book '{book_id}' not in books.yaml")
    pdf = ROOT / b["pdf"]
    # default page span per book
    if b.get("page_range"):
        lo, hi = b["page_range"]
    elif book_id == "vrikshayurveda":
        lo, hi = 1, 101          # full book; skip-logic drops the non-English pages
    elif book_id == "brihat_samhita":
        lo, hi = 275, 593        # the union span of the 12 wanted chapters
    else:
        lo, hi = 1, 999
    return pdf, lo, hi


def ocr_page(client, model, pdf, pg, poppler):
    from pdf2image import convert_from_path
    from google.genai import types

    kw = {"dpi": 220, "first_page": pg, "last_page": pg}
    if poppler:
        kw["poppler_path"] = poppler
    imgs = convert_from_path(str(pdf), **kw)
    if not imgs:
        return "", 0, 0, True

    # transient server errors (503/500/timeout) are common — retry with backoff
    # instead of crashing the whole run.
    from google.genai import errors as genai_errors
    resp = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model, contents=[PROMPT, imgs[0]],
                config=types.GenerateContentConfig(temperature=0.0),  # no token cap
            )
            break
        except genai_errors.ServerError as exc:
            wait = 5 * (attempt + 1)
            print(f"    page {pg}: server error ({exc.code}); retry {attempt+1}/5 in {wait}s")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 — network blips etc.
            wait = 5 * (attempt + 1)
            print(f"    page {pg}: {type(exc).__name__}; retry {attempt+1}/5 in {wait}s")
            time.sleep(wait)
    if resp is None:
        return "", 0, 0, True   # give up on this page -> treated as failed, not cached
    text = (resp.text or "").strip()
    u = getattr(resp, "usage_metadata", None)
    itok = int(getattr(u, "prompt_token_count", 0) or 0) if u else 0
    otok = int(getattr(u, "candidates_token_count", 0) or 0) if u else 0
    low = text.lower()
    is_error = (not text) or any(m in low for m in _ERROR_MARKERS)
    return text, itok, otok, is_error


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--pages", default=None, help="subrange like 41-93 (default: book's full span)")
    ap.add_argument("--throttle-seconds", type=float, default=1.5)
    args = ap.parse_args(argv)

    from google import genai

    pdf, lo, hi = book_pdf_and_pages(args.book)
    if args.pages:
        a, b = args.pages.split("-")
        lo, hi = int(a), int(b)
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")

    out_dir = ROOT / "corpus" / "raw_reocr" / args.book
    out_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=load_api_key())
    poppler = _resolve_poppler_path()

    print(f"Re-OCR {args.book}  pages {lo}..{hi}  model={args.model}")
    print(f"  -> writing to {out_dir}  (live corpus untouched)")
    ti = to = done = cached = skipped = failed = 0
    for pg in range(lo, hi + 1):
        cache = out_dir / f"page_{pg:04d}.txt"
        if cache.is_file():
            cached += 1
            continue
        text, i, o, is_error = ocr_page(client, args.model, pdf, pg, poppler)
        if is_error:
            # retry once, then leave UNcached so a re-run tries again
            time.sleep(4)
            text, i2, o2, is_error = ocr_page(client, args.model, pdf, pg, poppler)
            i += i2; o += o2
            if is_error:
                failed += 1
                print(f"  page {pg}: FAILED (not cached) — re-run to retry")
                ti += i; to += o
                time.sleep(args.throttle_seconds)
                continue
        payload = "" if text.strip() == "[SKIP]" else text
        cache.write_text(payload, encoding="utf-8")
        ti += i; to += o; done += 1
        if not payload:
            skipped += 1
        if done % 10 == 0 or pg == hi:
            print(f"  ...page {pg}: done={done} skip={skipped} fail={failed} "
                  f"| spend={cost_inr(ti,to):.2f} INR")
        if cost_inr(ti, to) > HARD_BUDGET_INR:
            print(f"HARD BUDGET {HARD_BUDGET_INR} INR hit — stopping."); break
        time.sleep(args.throttle_seconds)

    print("=" * 60)
    print(f"{args.book}: new={done} (skips={skipped}) cached={cached} failed={failed}")
    print(f"tokens in={ti:,} out={to:,} | spend={cost_inr(ti,to):.2f} INR "
          f"(${cost_inr(ti,to)/USD_TO_INR:.3f})")

    # External books (Krishi Parashara, Upavanavinoda) live as ONE merged .md
    # (status: ready_external). Assemble the per-page cache into that .md next to
    # the current one, suffixed .reocr.md so the live file stays untouched until
    # you compare and approve.
    cfg = yaml.safe_load((ROOT / "configs" / "corpus" / "books.yaml").read_text(encoding="utf-8"))
    bk = next((x for x in cfg["books"] if x["id"] == args.book), {})
    if bk.get("ocr_method") == "gemini_external" and bk.get("text_source"):
        parts = []
        for pg in range(lo, hi + 1):
            p = out_dir / f"page_{pg:04d}.txt"
            if p.is_file():
                body = p.read_text(encoding="utf-8").strip()
                if body:
                    parts.append(body)
        merged = "\n\n".join(parts).strip() + "\n"
        reocr_md = ROOT / (bk["text_source"] + ".reocr.md")
        reocr_md.parent.mkdir(parents=True, exist_ok=True)
        reocr_md.write_text(merged, encoding="utf-8")
        print(f"assembled merged -> {reocr_md}  ({len(merged)} chars)")
        print(f"compare with live: {bk['text_source']}")
    else:
        print(f"compare: diff  corpus/raw/{args.book}/  corpus/raw_reocr/{args.book}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
