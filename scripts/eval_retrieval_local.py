"""Local, Windows-safe Phase 11 RETRIEVAL evaluation — no Colab, no chromadb.

The Colab notebook keeps breaking on Python-3.13 dependency clashes. But the
retrieval half of Phase 11 needs neither a GPU LLM nor a network judge: it is
pure ranking maths over the corpus embeddings + the silver query set. This
script reproduces the HybridRetriever's exact pipeline in ONE process
(sentence-transformers + rank_bm25 + numpy — no chromadb, so the Windows
torch+chromadb DLL crash can't happen) and reports the same metric table the
notebook would, for the current 259-chunk corpus.

Faithful to src/rag/retriever.py:
  dense  = bge-large-en-v1.5, normalized cosine, top-20
  sparse = BM25Okapi over the same tokenizer, top-20
  fuse   = Reciprocal Rank Fusion, k=60
  rerank = bge-reranker-base cross-encoder, return top-5
Variants (src/eval/baselines.py): full, keyword_only, dense_only, hybrid_no_rerank.

Relevance is BOOK-LEVEL (silver labels have relevant_books, no chunk ids), so a
retrieved chunk counts as relevant iff its book_id is in the query's
relevant_books — exactly how the notebook scored it. Metrics come from the
project's own src/eval/retrieval_metrics.py so the numbers are directly
comparable to the earlier run.

Usage:  python scripts/eval_retrieval_local.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.retrieval_metrics import evaluate_retrieval  # exact project metrics

CHUNKS_DIR = ROOT / "corpus" / "chunks"
QUERY_SET = ROOT / "data" / "eval" / "silver_queries.json"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
K = 5
RRF_K = 60

_TOKEN_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def load_chunks() -> list[dict]:
    rows = []
    for jf in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda r: -r[1])]


def main() -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    from rank_bm25 import BM25Okapi

    chunks = load_chunks()
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    books = [c["book_id"] for c in chunks]
    id2book = dict(zip(ids, books))
    print(f"corpus: {len(chunks)} chunks across {len(set(books))} books")
    for b in sorted(set(books)):
        print(f"   {b:24} {books.count(b)}")

    qset = json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]
    answerable = [q for q in qset if q.get("expect_answerable")]
    print(f"queries: {len(qset)} ({len(answerable)} answerable, "
          f"{len(qset)-len(answerable)} negatives)\n")

    print(f"loading {EMBED_MODEL} (CPU) ...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    print("embedding corpus ...")
    doc_vecs = embedder.encode(texts, normalize_embeddings=True,
                               convert_to_numpy=True, batch_size=16,
                               show_progress_bar=True)
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    def dense_rank(q: str, n: int) -> list[str]:
        qv = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = doc_vecs @ qv
        order = np.argsort(-sims)[:n]
        return [ids[i] for i in order]

    def sparse_rank(q: str, n: int) -> list[str]:
        scores = bm25.get_scores(_tokenize(q))
        order = np.argsort(-scores)[:n]
        return [ids[i] for i in order]

    def rerank(q: str, cand_ids: list[str], k: int) -> list[str]:
        if not cand_ids:
            return []
        pairs = [[q, texts[ids.index(cid)]] for cid in cand_ids]
        scores = reranker.predict(pairs)
        order = np.argsort(-np.asarray(scores))[:k]
        return [cand_ids[i] for i in order]

    # variant -> ranked ids per query (only answerable queries have book labels)
    variants = ["full", "keyword_only", "dense_only", "hybrid_no_rerank"]
    tables = {}
    per_query_hits = {}   # for the coverage view (full variant)
    for v in variants:
        results = []
        for q in answerable:
            query = q["query"]
            relevant = {cid for cid in ids if id2book[cid] in set(q["relevant_books"])}
            if v == "keyword_only":
                ranked = sparse_rank(query, K)
            elif v == "dense_only":
                ranked = dense_rank(query, K)
            elif v == "hybrid_no_rerank":
                ranked = rrf_fuse([dense_rank(query, TOP_K_DENSE),
                                   sparse_rank(query, TOP_K_SPARSE)])[:K]
            else:  # full
                pool = rrf_fuse([dense_rank(query, TOP_K_DENSE),
                                 sparse_rank(query, TOP_K_SPARSE)])
                ranked = rerank(query, pool, K)
                per_query_hits[q["id"]] = {
                    "query": query[:55],
                    "relevant_books": q["relevant_books"],
                    "top_books": [id2book[c] for c in ranked],
                    "hit": bool(set(ranked) & relevant),
                }
            results.append((ranked, relevant))
        tables[v] = evaluate_retrieval(results, k=K).as_row()

    # -------- report --------
    print("\n" + "=" * 74)
    print(f"PHASE 11 RETRIEVAL METRICS  (corpus = {len(chunks)} chunks, 5 books)")
    print("=" * 74)
    hdr = f"{'variant':<20}{'P@5':>8}{'nDCG@5':>9}{'MRR':>8}{'Hit@5':>8}"
    print(hdr); print("-" * 74)
    for v in variants:
        r = tables[v]
        print(f"{v:<20}{r['P@5']:>8}{r['nDCG@5']:>9}{r['MRR']:>8}{r['Hit@5']:>8}")
    print("-" * 74)

    print("\nCOVERAGE (full variant) — did an answerable query retrieve a "
          "relevant-book passage?")
    hits = sum(1 for h in per_query_hits.values() if h["hit"])
    print(f"  {hits}/{len(per_query_hits)} answerable queries hit a relevant book "
          f"({100*hits/len(per_query_hits):.0f}%)")
    misses = [h for h in per_query_hits.values() if not h["hit"]]
    if misses:
        print("  misses:")
        for h in misses:
            print(f"    - '{h['query']}' wanted {h['relevant_books']} got {h['top_books']}")

    out = ROOT / "results" / "phase11_retrieval_local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_chunks": len(chunks), "per_book": {b: books.count(b) for b in sorted(set(books))},
         "variants": tables, "coverage": per_query_hits}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
