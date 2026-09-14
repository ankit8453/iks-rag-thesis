"""Does the NITI manual DISPLACE classical passages in the top-5? (§6j step 1)

The §6j hypothesis for the over-refusal regression (54.5% -> 72.7%) is that NITI
chunks are semantically adjacent enough to rank into the top-5 without being able
to answer a fungal-disease query, pushing classical passages out and starving the
generator of usable evidence.

This script tests that directly, with no Colab and no LLM:

  1. retrieve the top-5 for each answerable query over the FULL corpus (327 chunks)
  2. retrieve the top-5 again over the CLASSICAL-ONLY corpus (modern tier removed)
  3. report how many top-5 slots the modern tier took, and which classical passages
     were pushed out as a result

Retrieval mirrors src/rag/retriever.py exactly: dense (bge-large, top-20) + BM25
(top-20) -> Reciprocal Rank Fusion k=60 -> cross-encoder rerank -> top-5.

Windows-safe: no chromadb import anywhere (torch + chromadb in one process crashes),
scoring is pure numpy.

Usage:  python scripts/check_displacement.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHUNKS_DIR = ROOT / "corpus" / "chunks"
QUERY_SET = ROOT / "data" / "eval" / "silver_queries.json"
OUT_JSON = ROOT / "results" / "displacement_check.json"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
TOP_K_DENSE = TOP_K_SPARSE = 20
K = 5
RRF_K = 60
RERANK_POOL = 15   # cap the cross-encoder pool: on CPU it is the bottleneck

# books that are NOT classical treatises (see books.yaml source_tier)
MODERN_TIER = {"niti_natural_farming"}
# classical, but ingested later than the silver labels were written
LATE_CLASSICAL = {"kashyapiyakrishisukti"}

_TOKEN_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)


def _tok(t: str) -> list[str]:
    return [x.lower() for x in _TOKEN_RE.findall(t or "")]


def rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, i in enumerate(ranked, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda r: -r[1])]


def main() -> int:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder

    rows = []
    for jf in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    texts = [r["text"] for r in rows]
    books = [r["book_id"] for r in rows]
    n = len(rows)
    print(f"corpus: {n} chunks")
    for b, c in sorted(Counter(books).items()):
        tier = "MODERN" if b in MODERN_TIER else ("late-classical" if b in LATE_CLASSICAL else "classical")
        print(f"   {b:26} {c:4}   [{tier}]")

    qs = [q for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"] if q.get("expect_answerable")]
    print(f"queries: {len(qs)} answerable\n")

    print(f"loading {EMBED_MODEL} (CPU) ...")
    emb = SentenceTransformer(EMBED_MODEL, device="cpu")

    # Embedding 327 chunks on CPU takes ~9 min, so cache it. The key is the chunk
    # count + a hash of the ids, so the cache invalidates whenever the corpus changes.
    import hashlib
    key = hashlib.md5(("|".join(r["chunk_id"] for r in rows)).encode()).hexdigest()[:12]
    cache = ROOT / "results" / f"_docemb_{n}_{key}.npy"
    if cache.is_file():
        doc_v = np.load(cache)
        print(f"doc embeddings: loaded from cache ({cache.name})")
    else:
        doc_v = emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=16, show_progress_bar=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, doc_v)
        print(f"doc embeddings: computed and cached -> {cache.name}")
    bm25 = BM25Okapi([_tok(t) for t in texts])
    rr = CrossEncoder(RERANK_MODEL, device="cpu")

    qvecs = {q["id"]: emb.encode([q["query"]], normalize_embeddings=True,
                                 convert_to_numpy=True)[0] for q in qs}

    def top5(qid: str, query: str, allowed: np.ndarray) -> list[int]:
        """Full pipeline restricted to chunk indices in `allowed` (bool mask)."""
        qv = qvecs[qid]
        sims = doc_v @ qv
        sims_m = np.where(allowed, sims, -np.inf)
        dense = [int(i) for i in np.argsort(-sims_m)[:TOP_K_DENSE]]
        bs = bm25.get_scores(_tok(query))
        bs_m = np.where(allowed, bs, -np.inf)
        sparse = [int(i) for i in np.argsort(-bs_m)[:TOP_K_SPARSE]]
        pool = rrf([dense, sparse])[:RERANK_POOL]
        if not pool:
            return []
        scores = rr.predict([[query, texts[i]] for i in pool])
        order = np.argsort(-np.asarray(scores))[:K]
        return [pool[i] for i in order]

    all_mask = np.ones(n, dtype=bool)
    classical_mask = np.array([b not in MODERN_TIER for b in books])

    slot_books: Counter = Counter()
    displaced_total = 0
    per_query = []
    print("\nretrieving (2 passes per query: full corpus, then classical-only) ...")
    for qi, q in enumerate(qs, 1):
        print(f"  [{qi:2}/{len(qs)}] {q['id']} {(q.get('disease') or '')[:32]}", flush=True)
        full = top5(q["id"], q["query"], all_mask)
        cls = top5(q["id"], q["query"], classical_mask)
        full_books = [books[i] for i in full]
        slot_books.update(full_books)
        n_modern = sum(1 for b in full_books if b in MODERN_TIER)
        # classical passages that WOULD have been shown but were pushed out
        pushed = [i for i in cls if i not in full]
        displaced_total += len(pushed)
        per_query.append({
            "id": q["id"], "disease": q.get("disease") or "(unlabelled)", "query": q["query"][:70],
            "top5_books_full": full_books,
            "modern_in_top5": n_modern,
            "classical_pushed_out": [
                {"book": books[i], "verse": rows[i].get("verse_or_section"), "text": texts[i][:110]}
                for i in pushed
            ],
        })

    total_slots = len(qs) * K
    print("\n" + "=" * 78)
    print("WHO OCCUPIES THE TOP-5 SLOTS  (22 queries x 5 = 110 slots)")
    print("=" * 78)
    for b, c in slot_books.most_common():
        tag = "  <-- MODERN TIER" if b in MODERN_TIER else ""
        print(f"   {b:26} {c:4}  ({100*c/total_slots:5.1f}% of slots){tag}")

    n_modern_slots = sum(c for b, c in slot_books.items() if b in MODERN_TIER)
    q_with_modern = sum(1 for r in per_query if r["modern_in_top5"] > 0)
    print("\n" + "-" * 78)
    print(f"modern-tier slots taken     : {n_modern_slots}/{total_slots} ({100*n_modern_slots/total_slots:.1f}%)")
    print(f"queries with >=1 modern hit : {q_with_modern}/{len(qs)} ({100*q_with_modern/len(qs):.0f}%)")
    print(f"classical passages displaced: {displaced_total}  (would have been shown without the modern tier)")
    print("-" * 78)

    verdict = ("CONFIRMED - the modern tier is taking top-5 slots and pushing classical passages out"
               if n_modern_slots >= 0.15 * total_slots else
               "NOT CONFIRMED - the modern tier barely reaches the top-5; look elsewhere for the cause")
    print(f"\nVERDICT: {verdict}\n")

    print("per-query detail (only queries where the modern tier appears):")
    for r in per_query:
        if r["modern_in_top5"]:
            print(f"\n  {r['id']}  {r['disease']}")
            print(f"     top-5 books : {r['top5_books_full']}")
            for d in r["classical_pushed_out"]:
                print(f"     pushed out  : [{d['book']} {d['verse']}] {d['text']}...")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_chunks": n, "per_book": dict(Counter(books)),
        "slots": dict(slot_books), "total_slots": total_slots,
        "modern_slots": n_modern_slots, "queries_with_modern": q_with_modern,
        "classical_displaced": displaced_total, "verdict": verdict,
        "per_query": per_query,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
