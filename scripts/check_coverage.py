"""Can the texts actually answer the 22 evaluation queries? (§6m)

Across three corpus versions (206 -> 327 -> 233 chunks) the grounded-answer rate
has been frozen at exactly 13.64% = 3 of 22 queries, while over-refusal climbed
54.5% -> 72.7% -> 81.8%. Cleaning the corpus improved retrieval but not answers.
That points at a simpler explanation than any retrieval defect: **the classical
texts may genuinely not contain remedies for most of these diseases.**

The query set is SILVER — the project decided these 22 "should" be answerable;
nobody verified the texts can answer them. If they cannot, refusing is *correct*
and "over-refusal" is measuring the system against a false assumption.

This script retrieves the top-5 for each query (same pipeline as the system) and
prints the passages, with a heuristic remedy-language score, so a human can judge
what is actually there. The score only triages; the printed text is the evidence.

Local, no Colab, no LLM, no API cost. Embeddings cached to results/_docemb_*.npy.

Usage:  python scripts/check_coverage.py            # summary + verdicts
        python scripts/check_coverage.py --full     # also print passage text
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHUNKS_DIR = ROOT / "corpus" / "chunks"
QUERY_SET = ROOT / "data" / "eval" / "silver_queries.json"
OUT_JSON = ROOT / "results" / "coverage_check.json"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
TOP_K_DENSE = TOP_K_SPARSE = 20
RERANK_POOL = 15
K = 5
RRF_K = 60

# Words that signal an actual PRESCRIPTION, not just a mention of plants.
_ACTION = re.compile(
    r"\b(sprinkl\w*|smear\w*|water(?:ed|ing)?|appl\w+|treat\w*|cure[ds]?|"
    r"remov\w+|dust\w*|spray\w*|paste|decoction|mix\w+|pour\w*|bathe[ds]?|"
    r"fumigat\w*|smoke[ds]?|irrigat\w+|prune[ds]?|cut off)\b", re.I)
_MATERIAL = re.compile(
    r"\b(milk|ghee|honey|cow ?dung|kunapa|ash(?:es)?|sesame|turmeric|oil ?cake|"
    r"mustard|neem|vidanga|triphala|jaggery|buttermilk|urine|barley|salt)\b", re.I)
_SYMPTOM = re.compile(
    r"\b(disease[ds]?|disorder|spot[s]?|lesion|blight|rust|mildew|rot\b|wither\w*|"
    r"yellow\w*|pale\w*|dry\w*|insect[s]?|worm[s]?|pest[s]?|fung\w+|scorch\w*|"
    r"eaten|damage[ds]?)\b", re.I)

_TOKEN_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)


def _tok(t: str) -> list[str]:
    return [x.lower() for x in _TOKEN_RE.findall(t or "")]


def rrf(lists: list[list[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for ranked in lists:
        for rank, i in enumerate(ranked, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda r: -r[1])]


def remedy_score(text: str) -> tuple[int, int, int]:
    """(action verbs, remedy materials, symptom words) found in a passage."""
    return (len(set(m.group(0).lower() for m in _ACTION.finditer(text))),
            len(set(m.group(0).lower() for m in _MATERIAL.finditer(text))),
            len(set(m.group(0).lower() for m in _SYMPTOM.finditer(text))))


def main() -> int:
    show_full = "--full" in sys.argv
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
    qs = [q for q in json.loads(QUERY_SET.read_text(encoding="utf-8"))["queries"]
          if q.get("expect_answerable")]
    print(f"corpus {n} chunks | {len(qs)} answerable queries\n")

    emb = SentenceTransformer(EMBED_MODEL, device="cpu")
    key = hashlib.md5("|".join(r["chunk_id"] for r in rows).encode()).hexdigest()[:12]
    cache = ROOT / "results" / f"_docemb_{n}_{key}.npy"
    if cache.is_file():
        doc_v = np.load(cache)
        print(f"embeddings: cache hit ({cache.name})")
    else:
        print("embeddings: computing (one-time, ~10 min on CPU) ...")
        doc_v = emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=16, show_progress_bar=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, doc_v)
    bm25 = BM25Okapi([_tok(t) for t in texts])
    rr = CrossEncoder(RERANK_MODEL, device="cpu")

    results, tally = [], {"LIKELY": 0, "WEAK": 0, "NONE": 0}
    print("\nretrieving ...")
    for qi, q in enumerate(qs, 1):
        query = q["query"]
        qv = emb.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense = [int(i) for i in np.argsort(-(doc_v @ qv))[:TOP_K_DENSE]]
        sparse = [int(i) for i in np.argsort(-bm25.get_scores(_tok(query)))[:TOP_K_SPARSE]]
        pool = rrf([dense, sparse])[:RERANK_POOL]
        scores = rr.predict([[query, texts[i]] for i in pool])
        top = [pool[i] for i in np.argsort(-np.asarray(scores))[:K]]

        passages = []
        best = 0
        for i in top:
            a, m, s = remedy_score(texts[i])
            # a passage can plausibly answer if it PRESCRIBES something for a problem
            strength = (2 if (a >= 2 and m >= 1 and s >= 1) else
                        1 if (a >= 1 and (m >= 1 or s >= 2)) else 0)
            best = max(best, strength)
            passages.append({"book": books[i], "verse": rows[i].get("verse_or_section"),
                             "actions": a, "materials": m, "symptoms": s,
                             "strength": strength, "text": texts[i]})
        verdict = "LIKELY" if best == 2 else ("WEAK" if best == 1 else "NONE")
        tally[verdict] += 1
        results.append({"id": q["id"], "disease": q.get("disease") or "(unlabelled)",
                        "query": query, "verdict": verdict, "passages": passages})
        print(f"  [{qi:2}/{len(qs)}] {q['id']}  {verdict:6}  {(q.get('disease') or '')[:34]}")

    print("\n" + "=" * 76)
    print("CAN THE TEXTS ANSWER THESE QUERIES?  (judged from the retrieved passages)")
    print("=" * 76)
    for v, label in (("LIKELY", "a passage prescribes a treatment for a stated problem"),
                     ("WEAK", "related material, but no clear prescription"),
                     ("NONE", "nothing resembling a treatment")):
        print(f"  {v:7} {tally[v]:2}/{len(qs)}   {label}")
    print("=" * 76)
    print(f"\nThe system produced grounded answers for 3/22 (13.6%) in every run.")
    print(f"Passages that could support an answer: {tally['LIKELY']}/22 strong, "
          f"{tally['LIKELY'] + tally['WEAK']}/22 strong+weak.")
    if tally["LIKELY"] <= 6:
        print("\n=> Consistent with GENUINE LACK OF COVERAGE: refusing is largely correct,")
        print("   and 'over-refusal' is scoring the system against queries the texts")
        print("   cannot answer. The fix is more/better sources, not retrieval tuning.")
    else:
        print("\n=> The texts DO appear to support many more answers than the system gives.")
        print("   That points at the generator's sufficiency rule, not at coverage.")

    print("\n--- evidence: best passage per query ---")
    for r in results:
        b = max(r["passages"], key=lambda p: p["strength"])
        print(f"\n{r['id']} [{r['verdict']}] {r['disease']}")
        print(f"   query : {r['query'][:92]}")
        body = " ".join(b["text"].split())
        print(f"   best  : [{b['book']} {b['verse']}] act={b['actions']} mat={b['materials']} sym={b['symptoms']}")
        print(f"           {body[:(600 if show_full else 240)]}...")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"n_chunks": n, "tally": tally,
         "results": [{k: v for k, v in r.items() if k != "passages"} |
                     {"passages": [{k2: v2 for k2, v2 in p.items() if k2 != "text"} |
                                   {"text": p["text"][:400]} for p in r["passages"]]}
                     for r in results]}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
