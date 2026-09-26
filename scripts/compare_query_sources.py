"""Which writes better retrieval queries — Llama, or the hand-written evaluation set? (§6o)

Reads the two coverage runs and puts them side by side, per query. This is the comparison
Stage 0 exists for: until now the evaluation assumed the hand-written queries were the
right standard, and that was never checked against what the deployed bridge produces.

Inputs (produce them first):
    python scripts/check_coverage.py                                              -> silver_queries
    python scripts/check_coverage.py data/eval/llama_generated_queries.json llama -> llama

Usage:  python scripts/compare_query_sources.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "results" / "coverage_check_silver_queries.json"
LLAMA = ROOT / "results" / "coverage_check_llama.json"
OUT = ROOT / "results" / "query_source_comparison.md"

STRONG, MARGINAL = 0.35, 0.15
#: Below this, a difference in cross-encoder score is not worth interpreting on n=17.
NOISE = 0.05


def tier(score: float) -> str:
    return "strong" if score >= STRONG else ("marginal" if score >= MARGINAL else "none")


def load(path: Path) -> dict[str, dict]:
    if not path.is_file():
        print(f"missing: {path.relative_to(ROOT)}")
        return {}
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["results"]}


def main() -> int:
    hand, llama = load(HAND), load(LLAMA)
    if not hand or not llama:
        print("\nProduce both coverage runs first — see this file's docstring.")
        return 2

    gen = {q["id"]: q for q in json.loads(
        (ROOT / "data" / "eval" / "llama_generated_queries.json").read_text(encoding="utf-8")
    )["queries"]}

    shared = [q for q in hand if q in llama]
    rows = []
    for qid in sorted(shared):
        h = hand[qid]["top5"][0]["rerank_score"]
        l = llama[qid]["top5"][0]["rerank_score"]
        rows.append({
            "id": qid,
            "disease": hand[qid].get("disease", ""),
            "hand": h, "llama": l, "delta": l - h,
            "hand_q": hand[qid]["query"],
            "llama_q": gen.get(qid, {}).get("query", ""),
        })

    llama_wins = [r for r in rows if r["delta"] > NOISE]
    hand_wins = [r for r in rows if r["delta"] < -NOISE]
    ties = [r for r in rows if abs(r["delta"]) <= NOISE]

    print("=" * 78)
    print("WHO WRITES THE BETTER QUERY?   (top-1 cross-encoder score, higher is better)")
    print("=" * 78)
    print(f"  {'id':5} {'hand':>7} {'llama':>7} {'diff':>7}   winner    subject")
    for r in rows:
        w = "llama" if r["delta"] > NOISE else ("hand" if r["delta"] < -NOISE else "tie")
        print(f"  {r['id']:5} {r['hand']:7.4f} {r['llama']:7.4f} {r['delta']:+7.4f}   "
              f"{w:9} {r['disease'][:30]}")

    hm = sum(r["hand"] for r in rows) / len(rows)
    lm = sum(r["llama"] for r in rows) / len(rows)
    ht = {t: sum(1 for r in rows if tier(r["hand"]) == t) for t in ("strong", "marginal", "none")}
    lt = {t: sum(1 for r in rows if tier(r["llama"]) == t) for t in ("strong", "marginal", "none")}

    print("\n" + "=" * 78)
    print(f"  queries compared      {len(rows)}")
    print(f"  mean score            hand {hm:.4f}   llama {lm:.4f}   diff {lm - hm:+.4f}")
    print(f"  strong / marg / none  hand {ht['strong']}/{ht['marginal']}/{ht['none']}"
          f"      llama {lt['strong']}/{lt['marginal']}/{lt['none']}")
    print(f"  per-query wins        llama {len(llama_wins)}   hand {len(hand_wins)}   "
          f"tie {len(ties)}   (tie = within {NOISE})")
    print("=" * 78)

    if lm - hm > NOISE:
        verdict = ("LLAMA WRITES BETTER QUERIES. The hand-written evaluation set has been "
                   "understating the system: every retrieval number so far was measured on "
                   "queries weaker than the ones the deployed bridge actually sends. The "
                   "evaluation should switch to Llama's generated queries.")
    elif hm - lm > NOISE:
        verdict = ("THE HAND-WRITTEN QUERIES ARE BETTER. The deployed bridge is writing "
                   "weaker queries than it could, so real-world performance is worse than "
                   "the evaluation suggests. Fix the Strategy-B prompt, cheapest first: "
                   "few-shot examples taken from the corpus, then a corpus word list, then "
                   "two-round retrieval. Not fine-tuning.")
    else:
        verdict = ("NO MEANINGFUL DIFFERENCE. The bridge already writes queries as good as "
                   "hand-crafted ones, so the retrieval numbers are trustworthy and the "
                   "limit on usefulness really is corpus coverage, not query wording.")
    print("\nVERDICT: " + verdict)

    lines = ["# Llama's queries vs the hand-written evaluation set (§6o)", "",
             f"Compared on {len(rows)} disease-labelled queries, scored by the pipeline's own "
             "cross-encoder (top-1 relevance).", "",
             f"| | mean score | strong | marginal | none |",
             "|---|---|---|---|---|",
             f"| hand-written | {hm:.4f} | {ht['strong']} | {ht['marginal']} | {ht['none']} |",
             f"| Llama (Strategy B) | {lm:.4f} | {lt['strong']} | {lt['marginal']} | {lt['none']} |",
             "",
             f"Per-query: Llama better on {len(llama_wins)}, hand-written better on "
             f"{len(hand_wins)}, tied on {len(ties)} (tie = within {NOISE}).", "",
             f"**Verdict.** {verdict}", "",
             "---", "", "## Query by query", ""]
    for r in sorted(rows, key=lambda x: x["delta"]):
        lines += [f"**{r['id']}** — {r['disease']}  ·  hand {r['hand']:.4f} → "
                  f"llama {r['llama']:.4f} ({r['delta']:+.4f})",
                  f"- hand : _{r['hand_q']}_",
                  f"- llama: _{r['llama_q']}_", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
