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
DEFAULT_FP = ROOT / "corpus" / "_fingerprint_pre_citationfix.json"

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
    }


def label_report(rows: list[dict]) -> tuple[int, dict[str, int]]:
    labels = Counter(
        f"{r['book_id']}|ch.{r.get('chapter')}|v.{r.get('verse_or_section')}" for r in rows
    )
    collisions = {k: v for k, v in labels.items() if v > 1}
    return len(labels), collisions


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
    same_text = old["text_set_sha256"] == now["text_set_sha256"]
    print(f"\ncompared against {DEFAULT_FP.name}")
    print(f"  chunk count : {old['n_chunks']} -> {now['n_chunks']}")
    print(f"  total chars : {old['total_chars']:,} -> {now['total_chars']:,}")
    print(f"  text set    : {'IDENTICAL' if same_text else 'CHANGED'}")

    if same_text:
        print("\nPASS -- no chunk text was lost or altered; only metadata moved.")
        return 0

    # Text changed: say exactly how, so the cause is obvious.
    old_books, new_books = old["per_book"], now["per_book"]
    for b in sorted(set(old_books) | set(new_books)):
        o, n = old_books.get(b, 0), new_books.get(b, 0)
        if o != n:
            print(f"    {b}: {o} -> {n}")
    print("\nFAIL -- chunk text changed. If this was NOT intended, restore from backup:")
    print("  see corpus/RESTORE.md")
    print("If it WAS intended (e.g. front-matter stripping), store a new fingerprint:")
    print("  python scripts/verify_corpus.py --write <name>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
