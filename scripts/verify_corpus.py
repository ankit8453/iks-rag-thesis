"""Prove a corpus change did not lose or alter any content.

The citation-label fix (§6n) renames 17 Vrikshayurveda chunks whose
``verse_or_section`` labels collide. Renaming changes ``chunk_id`` — it is
``sha1(book|chapter|verse|text[:40])`` — so the corpus must be rebuilt. But the
*text* of every chunk must be untouched.

This script turns that into a checkable guarantee. It compares the current corpus
against a stored fingerprint of the sorted set of chunk texts:

    PASS -> every text is byte-identical; only metadata moved. Safe.
    FAIL -> content changed. Restore from backup (see corpus/RESTORE.md) before
            doing anything else.

It also reports label uniqueness, which is the point of the fix.

Usage:  python scripts/verify_corpus.py                     # compare to stored fingerprint
        python scripts/verify_corpus.py --write <name>       # store a new fingerprint
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "corpus" / "chunks"
#: The baseline is the most recent stored fingerprint that carries per-book hashes, so
#: a corpus change is always checked against the last verified state.
_FP_ORDER = ["_fingerprint_pre_vishvavallabha.json", "_fingerprint_pre_citationfix.json"]
DEFAULT_FP = next((ROOT / "corpus" / n for n in _FP_ORDER
                   if (ROOT / "corpus" / n).is_file()),
                  ROOT / "corpus" / _FP_ORDER[-1])

#: joined with a character that cannot occur in the text, so concatenation is unambiguous
_SEP = "␞"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(CHUNKS_DIR / "*.jsonl"))):
        rows += [json.loads(line) for line in Path(f).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    return rows


def fingerprint(rows: list[dict]) -> dict:
    texts = sorted(r["text"] for r in rows)
    return {
        "n_chunks": len(rows),
        "per_book": {b: sum(1 for r in rows if r["book_id"] == b)
                     for b in sorted({r["book_id"] for r in rows})},
        "text_set_sha256": hashlib.sha256(_SEP.join(texts).encode()).hexdigest(),
        "total_chars": sum(len(t) for t in texts),
        # per book, so adding a new book can be told apart from changing an old one
        "per_book_sha256": {
            b: hashlib.sha256(_SEP.join(sorted(r["text"] for r in rows
                                               if r["book_id"] == b)).encode()).hexdigest()
            for b in sorted({r["book_id"] for r in rows})
        },
    }


def label_report(rows: list[dict]) -> tuple[int, dict[str, int]]:
    labels = Counter(
        f"{r['book_id']}|ch.{r.get('chapter')}|v.{r.get('verse_or_section')}" for r in rows
    )
    collisions = {k: v for k, v in labels.items() if v > 1}
    return len(labels), collisions


def check_vector_db(live_ids: set[str]) -> int | None:
    """Compare ChromaDB against the chunk files. ``None`` if the store is unreadable.

    The chunk files are the source of truth, but *retrieval reads ChromaDB*, and the two
    can silently disagree: ``embed_chunks`` upserts by id, so a rebuild that renames a
    chunk ADDS the new id and leaves the old one orphaned. That is how a 233-chunk corpus
    came to hold 250 vectors after the citation fix (§6n) — 17 passages present twice, the
    second copy still carrying the ambiguous label the fix existed to remove. Checking only
    the chunk files reported PASS and missed it entirely.

    chromadb is imported here and torch is not: on Windows the two in one process give a
    silent 0xC0000005 or a cygrpc DLL failure.
    """
    try:
        import chromadb
    except Exception as exc:  # noqa: BLE001 - absence is not a corpus failure
        print(f"\nChromaDB not checked ({exc.__class__.__name__})")
        return None

    db = ROOT / "corpus" / "vector_db"
    if not db.is_dir():
        print("\nno vector store at corpus/vector_db - not checked")
        return None
    try:
        col = chromadb.PersistentClient(path=str(db)).get_collection("iks_corpus")
        db_ids = set(col.get()["ids"])
    except Exception as exc:  # noqa: BLE001
        print(f"\nChromaDB unreadable: {exc}")
        return None

    orphans = db_ids - live_ids
    absent = live_ids - db_ids
    print(f"\nvector store: {len(db_ids)} vectors for {len(live_ids)} chunks")
    if not orphans and not absent:
        print("  matches the chunk files exactly")
        return 0
    if orphans:
        print(f"  {len(orphans)} ORPHANED vector(s) - chunks that no longer exist.")
        print("  Retrieval would return stale duplicates of renamed passages. Fix first:")
        print("    python scripts/prune_stale_vectors.py --apply")
    if absent:
        print(f"  {len(absent)} chunk(s) NOT embedded - rebuild the corpus.")
    return 1


def main() -> int:
    if "--write" in sys.argv:
        idx = sys.argv.index("--write")
        name = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "current"
        out = ROOT / "corpus" / f"_fingerprint_{name}.json"
        out.write_text(json.dumps(fingerprint(load_rows()), indent=2), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
        return 0

    rows = load_rows()
    now = fingerprint(rows)

    print(f"corpus: {now['n_chunks']} chunks, {now['total_chars']:,} chars")
    for b, n in now["per_book"].items():
        print(f"  {b:24} {n:3}")

    n_labels, collisions = label_report(rows)
    print(f"\ncitation labels: {n_labels} distinct for {now['n_chunks']} chunks")
    if collisions:
        print(f"  {len(collisions)} label(s) shared by {sum(collisions.values())} chunks "
              f"-- a citation cannot identify a passage:")
        for k, v in sorted(collisions.items(), key=lambda kv: -kv[1]):
            print(f"    {k}  x{v}")
    else:
        print("  every chunk is uniquely citable")

    if not DEFAULT_FP.is_file():
        print(f"\nno stored fingerprint at {DEFAULT_FP.relative_to(ROOT)} -- nothing to compare")
        return 0

    old = json.loads(DEFAULT_FP.read_text(encoding="utf-8"))
    print(f"\ncompared against {DEFAULT_FP.name}")
    print(f"  chunk count : {old['n_chunks']} -> {now['n_chunks']}")
    print(f"  total chars : {old['total_chars']:,} -> {now['total_chars']:,}")

    db_status = check_vector_db({r["chunk_id"] for r in rows})

    if "per_book_sha256" in old:
        # Per book: every book in the baseline must be byte-identical; books not in the
        # baseline are ADDITIONS and are reported, not failed. This is what lets a new
        # treatise be ingested while proving the existing ones were not touched.
        changed, removed = [], []
        for b, h in old["per_book_sha256"].items():
            if b not in now["per_book_sha256"]:
                removed.append(b)
            elif now["per_book_sha256"][b] != h:
                changed.append(b)
        added = sorted(set(now["per_book_sha256"]) - set(old["per_book_sha256"]))
        for b in old["per_book_sha256"]:
            state = ("REMOVED" if b in removed else "CHANGED" if b in changed
                     else "identical")
            print(f"  {b:24} {state}")
        for b in added:
            print(f"  {b:24} ADDED ({now['per_book'][b]} chunks)")
        text_ok = not changed and not removed
    else:
        text_ok = old["text_set_sha256"] == now["text_set_sha256"]
        print(f"  text set    : {'IDENTICAL' if text_ok else 'CHANGED'}")

    if text_ok and db_status in (0, None):
        print("\nPASS -- no existing chunk text was lost or altered.")
        return 0
    if text_ok:
        print("\nFAIL -- chunk text is intact, but the vector store disagrees with the")
        print("chunk files (see above). Retrieval reads the vector store, not the files,")
        print("so fix that before running any evaluation.")
        return 1
    print("\nFAIL -- existing chunk text changed. If this was NOT intended, restore:")
    print("  see corpus/RESTORE.md")
    print("If it WAS intended, store a new fingerprint:")
    print("  python scripts/verify_corpus.py --write <name>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
