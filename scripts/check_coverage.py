"""Can the IKS corpus actually answer the 22 evaluation queries? (§6m)

The grounded-answer rate has been frozen at exactly 13.64% (3/22) across three
corpus versions (206 -> 327 -> 233 chunks), while over-refusal climbed to 81.8%.
Corpus expansion (§6j), NITI displacement (§6k) and the Brihat chapter-span bug
(§6l) have each been ruled out. The remaining hypothesis is that the texts simply
do not contain remedies for most of these diseases -- in which case refusing is
CORRECT and ``expect_answerable: true`` is the thing that is wrong.

This script does NOT try to judge that automatically. An earlier version scored
passages by counting action/material/symptom words anywhere in the chunk; because
every chunk is ~2800 characters of mixed content, that returned "answerable" for
22/22 -- a meaningless result. Word co-occurrence in a long chunk is not evidence.

So this version only gathers evidence for a human (or an LLM reading the dump):

1. retrieves top-5 per query with the real pipeline, keyed on ``chunk_id``;
2. reports how concentrated retrieval is (distinct chunks / rank-1 chunks);
3. reports citation-label collisions -- several chunks share one
   ``(book, chapter, verse)`` label, so a citation cannot identify a passage;
4. writes every DISTINCT retrieved chunk in full to a markdown dump, plus the
   per-query top-5 as chunk_id references, for direct reading.

Local, no Colab, no LLM, no API cost. Embeddings cached in results/_docemb_*.npy.

Usage:  python scripts/check_coverage.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHUNKS_DIR = ROOT / "corpus" / "chunks"
QUERY_SET = ROOT / "data" / "eval" / "silver_queries.json"
OUT_JSON = ROOT / "results" / "coverage_check.json"
OUT_DUMP = ROOT / "results" / "coverage_passages.md"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
TOP_K_DENSE = TOP_K_SPARSE = 20
RERANK_POOL = 15
K = 5
RRF_K = 60

_TOKEN_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)


def _tok(t: str) -> list[str]:
    return [x.lower() for x in _TOKEN_RE.findall(t or "")]


def rrf(lists: list[list[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for ranked in lists:
        for rank, i in enumerate(ranked, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda r: -r[1])]


def _label(r: dict) -> str:
    """The citation the generator is asked to emit for this chunk."""
    return f"{r['book_id']}, ch.{r.get('chapter')}, v.{r.get('verse_or_section')}"


def main() -> int:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder

    rows: list[dict] = []
    for jf in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    texts = [r["text"] for r in rows]
    n = len(rows)
    qs = [q for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]
          if q.get("expect_answerable")]
    print(f"corpus {n} chunks | {len(qs)} answerable queries")

    # --- citation labels: can a citation identify a passage at all? ----------
    labels = Counter(_label(r) for r in rows)
    collide = {k: v for k, v in labels.items() if v > 1}
    print(f"citation labels: {len(labels)} distinct for {n} chunks; "
          f"{len(collide)} labels shared by >1 chunk "
          f"({sum(collide.values())} chunks affected)")

    emb = SentenceTransformer(EMBED_MODEL, device="cpu")
    key = hashlib.md5("|".join(r["chunk_id"] for r in rows).encode()).hexdigest()[:12]
    cache = ROOT / "results" / f"_docemb_{n}_{key}.npy"
    if cache.is_file():
        doc_v = np.load(cache)
        print(f"embeddings: cache hit ({cache.name})")
    else:
        print("embeddings: computing (one-time, ~8 min on CPU) ...")
        doc_v = emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=16, show_progress_bar=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, doc_v)
    bm25 = BM25Okapi([_tok(t) for t in texts])
    rr = CrossEncoder(RERANK_MODEL, device="cpu")

    results = []
    print("\nretrieving ...")
    for qi, q in enumerate(qs, 1):
        query = q["query"]
        qv = emb.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense = [int(i) for i in np.argsort(-(doc_v @ qv))[:TOP_K_DENSE]]
        sparse = [int(i) for i in np.argsort(-bm25.get_scores(_tok(query)))[:TOP_K_SPARSE]]
        pool = rrf([dense, sparse])[:RERANK_POOL]
        scores = rr.predict([[query, texts[i]] for i in pool])
        order = np.argsort(-np.asarray(scores))[:K]
        top = [(pool[int(i)], float(scores[int(i)])) for i in order]
        results.append({
            "id": q["id"], "disease": q.get("disease") or "(unlabelled)",
            "query": query,
            "top5": [{"chunk_id": rows[i]["chunk_id"], "book": rows[i]["book_id"],
                      "label": _label(rows[i]), "rerank_score": round(s, 4)}
                     for i, s in top],
        })
        print(f"  [{qi:2}/{len(qs)}] {q['id']}  {(q.get('disease') or '')[:38]}")

    # --- concentration, keyed on chunk_id (labels are not unique) ------------
    top1 = Counter(r["top5"][0]["chunk_id"] for r in results)
    top5 = Counter(p["chunk_id"] for r in results for p in r["top5"])
    by_id = {r["chunk_id"]: r for r in rows}

    print("\n" + "=" * 74)
    print("RETRIEVAL CONCENTRATION  (how varied is the evidence the LLM sees?)")
    print("=" * 74)
    print(f"  distinct chunks across all top-5 : {len(top5)} of {n}")
    print(f"  distinct rank-1 chunks           : {len(top1)} for {len(qs)} queries")
    print(f"  top-5 slots                      : {sum(top5.values())}")
    print("\n  most-retrieved chunks:")
    print("  chunk_id          book                 top1 top5  label")
    for cid, c in top5.most_common(10):
        r = by_id[cid]
        print(f"  {cid[:16]}  {r['book_id'][:20]:20} {top1[cid]:4} {c:4}  "
              f"ch.{r.get('chapter')} v.{r.get('verse_or_section')}")

    # --- dump distinct chunks in full, for reading ---------------------------
    lines = ["# Retrieved passages — coverage check (§6m)", "",
             f"{len(top5)} distinct chunks fill {sum(top5.values())} top-5 slots "
             f"across {len(qs)} queries.", "",
             "Question to answer while reading: **does this passage prescribe a "
             "treatment for a plant problem?** (not: does it mention plants)", "",
             "---", "", "## Per-query top-5", ""]
    for r in results:
        lines.append(f"**{r['id']}** — {r['disease']} — _{r['query']}_")
        for rank, p in enumerate(r["top5"], 1):
            lines.append(f"  {rank}. `{p['chunk_id'][:16]}` {p['label']} "
                         f"(score {p['rerank_score']})")
        lines.append("")
    lines += ["---", "", "## Distinct chunks, full text", ""]
    for cid, c in top5.most_common():
        r = by_id[cid]
        lines += [f"### `{cid[:16]}` — {r['book_id']} ch.{r.get('chapter')} "
                  f"v.{r.get('verse_or_section')}",
                  f"_retrieved in {c} top-5 slots; rank-1 for {top1[cid]} queries; "
                  f"{len(r['text'])} chars_", "",
                  " ".join(r["text"].split()), ""]
    OUT_DUMP.parent.mkdir(parents=True, exist_ok=True)
    OUT_DUMP.write_text("\n".join(lines), encoding="utf-8")

    OUT_JSON.write_text(json.dumps({
        "n_chunks": n, "n_queries": len(qs),
        "distinct_chunks_in_top5": len(top5), "distinct_rank1_chunks": len(top1),
        "colliding_citation_labels": collide,
        "chunk_top5_counts": dict(top5.most_common()),
        "results": results}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT_JSON.relative_to(ROOT)}")
    print(f"saved -> {OUT_DUMP.relative_to(ROOT)}  (read this to judge coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
