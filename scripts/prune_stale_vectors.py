"""Delete ChromaDB vectors whose chunk no longer exists (§6n).

Why this is needed. ``chunk_id`` is ``sha1(book|chapter|verse|text[:40])``, so the
citation-label fix gave 17 Vrikshayurveda chunks new ids. ``embed_chunks`` upserts by id,
which adds the new entries but **cannot know the old ones are now orphans** — so the
collection ended up holding 250 vectors for a 233-chunk corpus.

That is not harmless. The 17 orphans carry the *same text* as their replacements, so
retrieval would return the same passage twice inside one top-5, and the old ambiguous
labels (``v.1.2`` pointing at five passages) would reappear in citations — undoing the fix
this rebuild was for.

Deleting them is safe: every orphan's text is already present under its new id, which
``--check`` verifies before anything is removed.

chromadb is imported here and torch is not: on Windows the two in one process give a
silent 0xC0000005 or a cygrpc DLL failure.

Usage:  python scripts/prune_stale_vectors.py            # report only
        python scripts/prune_stale_vectors.py --apply    # delete the orphans
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "corpus" / "vector_db"
COLLECTION = "iks_corpus"


def live_chunks() -> dict[str, str]:
    """chunk_id -> text, from the chunk files, which are the source of truth."""
    out: dict[str, str] = {}
    for f in glob.glob(str(ROOT / "corpus" / "chunks" / "*.jsonl")):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["chunk_id"]] = row["text"]
    return out


def main() -> int:
    import chromadb

    apply = "--apply" in sys.argv
    client = chromadb.PersistentClient(path=str(DB_PATH))
    col = client.get_collection(COLLECTION)

    got = col.get(include=["metadatas", "documents"])
    db_ids = list(got["ids"])
    db_docs = dict(zip(db_ids, got["documents"]))
    db_meta = dict(zip(db_ids, got["metadatas"]))

    live = live_chunks()
    stale = sorted(set(db_ids) - set(live))
    missing = sorted(set(live) - set(db_ids))

    print(f"ChromaDB      : {col.count()} vectors")
    print(f"chunk files   : {len(live)} chunks")
    print(f"orphans       : {len(stale)}")
    print(f"not embedded  : {len(missing)}")

    if missing:
        print("\nSome chunks are missing from ChromaDB. Prune nothing — rebuild instead:")
        for i in missing[:10]:
            print(f"  {i[:16]}")
        return 1

    if not stale:
        print("\nNothing to prune; ChromaDB matches the chunk files exactly.")
        return 0

    # Safety gate: an orphan may only be deleted if its text survives under another id.
    live_texts = set(live.values())
    unsafe = [i for i in stale if db_docs.get(i) not in live_texts]

    print("\norphaned vectors:")
    for i in stale:
        m = db_meta.get(i, {})
        safe = "text kept under new id" if db_docs.get(i) in live_texts else "TEXT NOT FOUND"
        print(f"  {i[:16]}  {m.get('book_id','?'):16} "
              f"ch.{m.get('chapter','?')} v.{m.get('verse_or_section','?'):8}  {safe}")

    if unsafe:
        print(f"\nREFUSING TO DELETE: {len(unsafe)} orphan(s) hold text that exists nowhere "
              "else. Deleting them would lose content. Investigate before proceeding.")
        return 1

    print(f"\nAll {len(stale)} orphans duplicate text already present under a new id, so "
          "deleting them removes no content.")

    if not apply:
        print("\nreport only — re-run with --apply to delete")
        return 0

    col.delete(ids=stale)
    after = col.count()
    print(f"\ndeleted {len(stale)}; ChromaDB now holds {after} vectors")
    if after != len(live):
        print(f"WARNING: expected {len(live)}. Investigate before running any evaluation.")
        return 1
    print("ChromaDB now matches the chunk files exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
