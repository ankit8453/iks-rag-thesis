"""OCR a scanned PDF, page by page, with a persistent per-page cache.

Both classical-text PDFs in Phase 3 are image-only (no text layer), so
we rasterise each page via :mod:`pdf2image` (poppler) and run
:mod:`pytesseract` over the resulting PIL images. Tesseract is invoked
with ``--oem 1 --psm 6`` and ``lang=eng`` per Locked Decision #1.

Per-page cache
--------------

Every OCR'd page is written to ``corpus/raw/<book_id>/page_<NNNN>.txt``
the first time it's processed. On re-runs, pages with existing cache
files are loaded from disk and Tesseract is NOT re-invoked. This keeps
incremental development cheap: a typo fix in chunking doesn't re-OCR
700+ pages.

Windows binary discovery
------------------------

Tesseract and Poppler binaries are auto-discovered at common Windows
install locations if ``which`` doesn't find them on ``PATH``. Both can
also be overridden via the ``TESSERACT_CMD`` and ``POPPLER_PATH``
environment variables.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logging_setup import get_logger
from src.utils.paths import CORPUS_RAW_DIR

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

_LOGGER = get_logger(__name__)

DEFAULT_DPI: int = 300
DEFAULT_TESSERACT_CONFIG: str = "--oem 1 --psm 6"
DEFAULT_TESSERACT_LANG: str = "eng"


_WINDOWS_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

_WINDOWS_POPPLER_CANDIDATES = (
    r"C:\poppler\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
)

# Globs to find versioned poppler-windows installs like ``C:\poppler-26.02.0\Library\bin``.
_WINDOWS_POPPLER_GLOBS = (
    r"C:\poppler-*\Library\bin",
    r"C:\Program Files\poppler-*\Library\bin",
    r"C:\Program Files (x86)\poppler-*\Library\bin",
)


@dataclass
class PageText:
    """One PDF page's raw OCR output, addressed by 1-based page number."""

    pdf_page_number: int
    raw_text: str


def _resolve_tesseract_cmd() -> str | None:
    """Find ``tesseract`` on PATH, in $TESSERACT_CMD, or at known Windows paths."""
    env = os.environ.get("TESSERACT_CMD", "").strip()
    if env and Path(env).is_file():
        return env

    found = shutil.which("tesseract")
    if found:
        return found

    for candidate in _WINDOWS_TESSERACT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _resolve_poppler_path() -> str | None:
    """Find poppler binaries directory.

    Returns ``None`` if poppler is on PATH (pdf2image will then find
    ``pdftoppm`` on its own); returns the directory path otherwise.

    Discovery order: ``POPPLER_PATH`` env var → ``PATH`` → fixed Windows
    candidates → versioned globs like ``C:\\poppler-*\\Library\\bin``
    (latest version wins).
    """
    import glob  # noqa: PLC0415

    env = os.environ.get("POPPLER_PATH", "").strip()
    if env and Path(env).is_dir():
        return env

    if shutil.which("pdftoppm"):
        return None  # pdf2image will find it without a hint

    for candidate in _WINDOWS_POPPLER_CANDIDATES:
        if Path(candidate).is_dir() and (Path(candidate) / "pdftoppm.exe").is_file():
            return candidate

    # Versioned installs — sort so the highest version wins.
    versioned: list[Path] = []
    for pattern in _WINDOWS_POPPLER_GLOBS:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_dir() and (p / "pdftoppm.exe").is_file():
                versioned.append(p)
    if versioned:
        versioned.sort()
        return str(versioned[-1])
    return None


def _ensure_pytesseract() -> None:
    """Configure :mod:`pytesseract` to point at the Tesseract binary."""
    import pytesseract  # noqa: PLC0415

    cmd = _resolve_tesseract_cmd()
    if cmd is None:
        raise RuntimeError(
            "Tesseract binary not found. Install via UB-Mannheim "
            "(`tesseract-ocr-w64-setup-*.exe`) on Windows or "
            "`apt-get install tesseract-ocr` on Linux. Optionally set "
            "the TESSERACT_CMD environment variable to the exe path."
        )
    pytesseract.pytesseract.tesseract_cmd = cmd
    _LOGGER.info("Using Tesseract at %s", cmd)


def _page_cache_path(book_id: str, page_number: int) -> Path:
    return CORPUS_RAW_DIR / book_id / f"page_{page_number:04d}.txt"


def _load_cached_page(book_id: str, page_number: int) -> str | None:
    cache = _page_cache_path(book_id, page_number)
    if cache.is_file():
        return cache.read_text(encoding="utf-8")
    return None


def _save_cached_page(book_id: str, page_number: int, text: str) -> None:
    cache = _page_cache_path(book_id, page_number)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")


def _ocr_one_image(image: "PILImage") -> str:
    import pytesseract  # noqa: PLC0415

    return pytesseract.image_to_string(
        image, lang=DEFAULT_TESSERACT_LANG, config=DEFAULT_TESSERACT_CONFIG,
    )


def ocr_pdf(
    pdf_path: Path | str,
    book_id: str,
    *,
    dpi: int = DEFAULT_DPI,
    page_range: tuple[int, int] | None = None,
) -> list[PageText]:
    """OCR a PDF page-by-page with persistent caching.

    Parameters
    ----------
    pdf_path
        Path to the source PDF. Must exist.
    book_id
        Stable identifier from ``configs/corpus/books.yaml`` — used as
        the cache subdirectory name under ``corpus/raw/``.
    dpi
        Rasterisation DPI. 300 is the locked default.
    page_range
        Optional ``(start, end)`` inclusive 1-based page range. ``None``
        means the whole document.

    Returns
    -------
    list[PageText]
        One entry per processed page, in document order.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    _ensure_pytesseract()
    poppler_path = _resolve_poppler_path()
    if poppler_path is None and not shutil.which("pdftoppm"):
        raise RuntimeError(
            "Poppler binaries not found. Install poppler-windows from "
            "https://github.com/oschwartz10612/poppler-windows/releases "
            "and add its `Library\\bin` directory to PATH, or set the "
            "POPPLER_PATH environment variable to that directory. "
            "On Linux: `apt-get install poppler-utils`."
        )

    _LOGGER.info("OCR start: %s (dpi=%d, book_id=%s)", pdf_path.name, dpi, book_id)
    from pdf2image import convert_from_path  # noqa: PLC0415

    # First pass: get the document's page count via a thin import, then
    # walk page-by-page so we can SKIP cached pages without rasterising
    # them. ``convert_from_path`` rasterises (slow!), so we only call it
    # for uncached pages.
    from pdf2image.pdf2image import pdfinfo_from_path  # noqa: PLC0415

    info_kwargs = {"poppler_path": poppler_path} if poppler_path else {}
    info = pdfinfo_from_path(str(pdf_path), **info_kwargs)
    n_pages = int(info["Pages"])
    if page_range is None:
        first, last = 1, n_pages
    else:
        first, last = page_range
        first = max(1, first)
        last = min(n_pages, last)

    _LOGGER.info("PDF has %d pages; processing %d..%d", n_pages, first, last)

    results: list[PageText] = []
    cache_hits = 0
    for page_number in range(first, last + 1):
        cached = _load_cached_page(book_id, page_number)
        if cached is not None:
            results.append(PageText(pdf_page_number=page_number, raw_text=cached))
            cache_hits += 1
            continue

        # Rasterise ONLY this page and OCR it.
        kwargs = {
            "dpi": dpi,
            "first_page": page_number,
            "last_page": page_number,
        }
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(str(pdf_path), **kwargs)
        if not images:
            _LOGGER.warning("pdf2image returned 0 images for page %d", page_number)
            text = ""
        else:
            text = _ocr_one_image(images[0])
        _save_cached_page(book_id, page_number, text)
        results.append(PageText(pdf_page_number=page_number, raw_text=text))

        if page_number % 25 == 0 or page_number == last:
            _LOGGER.info(
                "  ocr progress: page %d/%d (cache_hits=%d)",
                page_number, last, cache_hits,
            )

    _LOGGER.info(
        "OCR done: %s -> %d pages (cache_hits=%d, new=%d)",
        pdf_path.name, len(results), cache_hits, len(results) - cache_hits,
    )
    return results


__all__ = [
    "DEFAULT_DPI",
    "PageText",
    "ocr_pdf",
]
