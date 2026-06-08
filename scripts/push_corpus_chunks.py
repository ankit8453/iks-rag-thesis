"""Push Phase 3 corpus chunks to a private HF Hub dataset (Phase 7 PART 1).

Phase 3 built the IKS corpus locally: 78 Vrikshayurveda chunks plus 207
Brihat-Samhita-12-chapter chunks land in
``corpus/chunks/{vrikshayurveda,brihat_samhita}.jsonl``. Phase 7 runs on
Colab (single-process Linux to dodge the Windows
torch+chromadb DLL conflict) and rebuilds ChromaDB in-session from
those same chunks. This script is the transport: it pushes the merged
285-row corpus to ``ankit-iiitdmj/iks-corpus-chunks`` as a **private**
HF dataset (master plan §38 — the translation text is copyrighted, so
it never travels via the public GitHub repo).

Schema validation
-----------------

Every JSONL row must carry the full Phase 3 metadata: ``source_text``,
``edition``, ``chapter``, ``verse_or_section``, ``topic_tags``,
``original_language``, ``translator``, ``chunk_id``, ``text``. The
``book_id`` column is added here so downstream code can split / filter
without re-parsing ``chunk_id``.

Idempotency
-----------

``Dataset.push_to_hub`` overwrites the existing snapshot when called
with the same repo id, so re-running this script after a Phase 3
re-build is safe — chunk IDs are deterministic sha1 hashes so the
collection stays in sync.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import CORPUS_CHUNKS_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

EXPECTED_HF_USERNAME = "ankit-iiitdmj"
TARGET_REPO = f"{EXPECTED_HF_USERNAME}/iks-corpus-chunks"

# Sources to merge into the single uploaded dataset, in this order. New
# books from Phase 3's pending list (Krishi Parashara, Upavanavinoda,
# Kashyapiyakrishisukti, sixth-text-TBD) will append to this list as
# their JSONL files land in ``corpus/chunks/``.
BOOK_FILES: dict[str, str] = {
    "vrikshayurveda": "vrikshayurveda.jsonl",
    "brihat_samhita": "brihat_samhita.jsonl",
    "krishi_parashara": "krishi_parashara.jsonl",
    "upavanavinoda": "upavanavinoda.jsonl",
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "source_text",
    "edition",
    "chapter",
    "verse_or_section",
    "topic_tags",
    "original_language",
    "translator",
    "chunk_id",
    "text",
)


def _preflight_auth() -> None:
    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().whoami()
    actual = info.get("name")
    if actual != EXPECTED_HF_USERNAME:
        raise PermissionError(
            f"HF Hub token belongs to '{actual}', expected "
            f"'{EXPECTED_HF_USERNAME}'. Run `huggingface-cli login` with "
            f"the correct Write token before re-running."
        )
    token = info.get("auth", {}).get("accessToken", {})
    if token.get("role") != "write":
        raise PermissionError(
            f"HF Hub token role is '{token.get('role')}', need 'write'."
        )
    _LOGGER.info("HF Hub pre-flight ok: user=%s role=write", actual)


def _load_book(book_id: str, jsonl_name: str) -> list[dict]:
    path = CORPUS_CHUNKS_DIR / jsonl_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Chunk JSONL missing for book {book_id!r}: {path}. "
            f"Re-run `python -m src.rag.corpus.build_corpus` first."
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no} — JSON decode error: {exc}"
                ) from exc
            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise RuntimeError(
                    f"{path}:{line_no} — row missing required fields: {missing}. "
                    f"Row keys present: {sorted(row)}"
                )
            row["book_id"] = book_id
            # ``topic_tags`` is a Python list; normalise to the same str
            # representation Phase 3 used in Chroma metadata so the two
            # sources stay byte-for-byte compatible.
            if isinstance(row["topic_tags"], list):
                row["topic_tags"] = ", ".join(str(t) for t in row["topic_tags"])
            rows.append(row)
    _LOGGER.info("Loaded %d rows from %s", len(rows), path.name)
    return rows


def _build_dataset(all_rows: list[dict]):
    """Wrap merged rows in a ``datasets.Dataset`` ready for push_to_hub."""
    from datasets import Dataset, Features, Value  # noqa: PLC0415

    # Lock the column dtypes so a row with subtly different types
    # (e.g. integer chapter vs string chapter) is caught at build time,
    # not after the upload starts.
    features = Features(
        {
            "book_id": Value("string"),
            "chunk_id": Value("string"),
            "source_text": Value("string"),
            "edition": Value("string"),
            "chapter": Value("string"),
            "verse_or_section": Value("string"),
            "topic_tags": Value("string"),
            "original_language": Value("string"),
            "translator": Value("string"),
            "text": Value("string"),
        }
    )
    # Drop any extras (e.g. ``metadata_extras`` carried by chunking.py)
    # — they aren't part of the public dataset schema.
    keep_cols = list(features)
    rows_lean = [{k: str(r.get(k, "")) for k in keep_cols} for r in all_rows]
    return Dataset.from_list(rows_lean, features=features)


def _dataset_card(total: int, per_book: dict[str, int]) -> str:
    per_book_md = "\n".join(f"- `{book}`: {n} chunks" for book, n in per_book.items())
    return (
        "---\n"
        "task_categories:\n  - text-retrieval\n"
        "size_categories:\n  - n<1K\n"
        "---\n\n"
        f"# {TARGET_REPO}\n\n"
        "Indian Knowledge Systems (IKS) classical-text corpus chunks for the "
        "Phase 7 grounded-RAG pipeline. Chunks are 200–500-token verse / passage "
        "units extracted by the Phase 3 pipeline at "
        "`src/rag/corpus/build_corpus.py` from scanned PDFs of four classical "
        "Sanskrit treatises in English translation: **Vrikshayurveda** "
        "(Surapala, tr. Sadhale, AAHF 1996), **Brihat Samhita Part 1** "
        "(Varahamihira, tr. Bhat, MLBD; 12 selected chapters), **Krishi "
        "Parashara** (tr. Majumdar & Banerji, Bibliotheca Indica 1960), and "
        "**Upavanavinoda** (Sarngadhara, tr. Majumdar, IRI Indian Positive "
        "Sciences). OCR was performed with Gemini 3.5 Flash (Phase 3b.2) for "
        "higher fidelity than Tesseract.\n\n"
        "**Private** — chunks contain copyrighted translation text (AAHF / MLBD "
        "/ Bibliotheca Indica / IRI editions); master plan §38 forbids "
        "redistributing them via the public GitHub repo. They live here so "
        "Colab notebooks can rebuild ChromaDB in-session without depending on "
        "the laptop's local store.\n\n"
        f"## Counts\n\n- total: {total}\n{per_book_md}\n\n"
        "## Columns\n\n"
        "- `book_id` — `vrikshayurveda` / `brihat_samhita` / `krishi_parashara` / `upavanavinoda` (extensible)\n"
        "- `chunk_id` — deterministic sha1 of (book|chapter|verse|first40chars)\n"
        "- `source_text` — canonical book title\n"
        "- `edition` — translation edition (e.g. \"Asian Agri-History Foundation, 1996\")\n"
        "- `chapter` — string; `'full'` for full-book scope, numeric for chapter-scope\n"
        "- `verse_or_section` — verse marker or sequential `section_N`\n"
        "- `topic_tags` — comma-joined topical tags from the book's YAML entry\n"
        "- `original_language` — typically `Sanskrit`\n"
        "- `translator` — full name\n"
        "- `text` — cleaned English passage (post-OCR, post-Devanagari-drop)\n"
    )


def main() -> int:
    _preflight_auth()

    all_rows: list[dict] = []
    per_book: dict[str, int] = {}
    for book_id, jsonl_name in BOOK_FILES.items():
        rows = _load_book(book_id, jsonl_name)
        all_rows.extend(rows)
        per_book[book_id] = len(rows)

    total = len(all_rows)
    _LOGGER.info("Validated %d rows across %d books.", total, len(per_book))

    dataset = _build_dataset(all_rows)
    _LOGGER.info("Built datasets.Dataset of %d rows.", len(dataset))

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(
        repo_id=TARGET_REPO, repo_type="dataset", private=True, exist_ok=True,
    )

    _LOGGER.info("Pushing %d rows to %s ...", total, TARGET_REPO)
    dataset.push_to_hub(TARGET_REPO, private=True)
    _LOGGER.info("Push complete.")

    card = _dataset_card(total, per_book)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=TARGET_REPO,
        repo_type="dataset",
    )
    _LOGGER.info("README pushed.")

    print()
    print(f"Pushed {total} rows to https://huggingface.co/datasets/{TARGET_REPO}")
    for book, n in per_book.items():
        print(f"  {book}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
