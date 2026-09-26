"""Merge measured candidate queries into the silver evaluation set (§6n).

The candidates in ``data/eval/new_queries_draft_*.json`` were written from the
domain, deliberately without consulting the corpus, so their answerability is
unknown until measured. ``scripts/check_coverage.py`` scores each one with the
pipeline's own cross-encoder; this script reads that score file and writes the
label, using the same thresholds as §6m:

    >= 0.35  strong    -> expect_answerable = true
    0.15-0.35 marginal -> expect_answerable = true, flagged as marginal
    < 0.15   none      -> expect_answerable = false, kind "no_coverage"

The point is that the label follows the evidence rather than an assumption. That
assumption is exactly what made "over-refusal 81.8%" meaningless for three
evaluation cycles.

Candidates that score "none" are KEPT, not deleted. Deleting them would leave a
query set biased toward what the corpus happens to contain, which would inflate
every future number.

Run ``check_coverage.py`` on the draft file first:

    python scripts/check_coverage.py data/eval/new_queries_draft_2026-09-26.json newq
    python scripts/merge_new_queries.py            # dry run, prints the plan
    python scripts/merge_new_queries.py --apply
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "eval" / "silver_queries.json"
DRAFT = ROOT / "data" / "eval" / "new_queries_draft_2026-09-26.json"
SCORES = ROOT / "results" / "coverage_check_newq.json"

STRONG, MARGINAL = 0.35, 0.15

#: How a reading-based grade becomes an evaluation label.
#:
#: The grade, not the cross-encoder score, decides this. The score marked all 16 candidates
#: "strong" (up to 0.987) while several top-1 passages were plainly off-topic, because it
#: measures vocabulary overlap: good evidence of ABSENCE, unreliable evidence of PRESENCE.
#: "partial" is kept answerable but flagged, so a refusal there is not scored as an error.
GRADE_TO_LABEL = {
    "answers": ("strong", True),
    "partial": ("marginal", True),
    "no": ("none", False),
}


def tier_of(score: float) -> str:
    """Score-only tier. Retained for reporting the score/grade disagreement."""
    return "strong" if score >= STRONG else ("marginal" if score >= MARGINAL else "none")


def main() -> int:
    apply = "--apply" in sys.argv
    for f in (SILVER, DRAFT, SCORES):
        if not f.is_file():
            print(f"missing: {f.relative_to(ROOT)}")
            if f is SCORES:
                print("\nRun the measurement first:")
                print("  python scripts/check_coverage.py "
                      f"{DRAFT.relative_to(ROOT)} newq")
            return 2

    silver = json.loads(SILVER.read_text(encoding="utf-8"))
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    scored = {r["id"]: r for r in json.loads(SCORES.read_text(encoding="utf-8"))["results"]}
    existing = {q["id"] for q in silver["queries"]}

    plan, tally = [], {"strong": 0, "marginal": 0, "none": 0}
    for cand in draft["queries"]:
        qid = cand["id"]
        if qid in existing:
            print(f"  {qid}: already in the silver set, skipped")
            continue
        row = scored.get(qid)
        if row is None:
            print(f"  {qid}: NOT SCORED — excluded (never add an unmeasured query)")
            continue
        score = row["top5"][0]["rerank_score"]
        grade = cand.get("human_grade")
        if grade not in GRADE_TO_LABEL:
            print(f"  {qid}: no human_grade — excluded. Grade it by reading the passages "
                  f"in results/coverage_passages_newq.md first.")
            continue
        tier, answerable = GRADE_TO_LABEL[grade]
        score_tier = tier_of(score)
        tally[tier] += 1
        books = sorted({p["book"] for p in row["top5"][:3]})
        q = {
            "id": qid,
            "crop": "general",
            "disease": None,
            "domain": cand["domain"],
            "query": cand["query"],
            "relevant_books": books,
            "relevant_chunk_ids": [],
            "expect_answerable": answerable,
            "coverage_tier": tier,
            "human_grade": grade,
            "grade_reason": cand.get("grade_reason", ""),
            "top1_rerank_score": score,
            "score_only_tier": score_tier,
            "added": "2026-09-26",
        }
        if tier == "none":
            q["unanswerable_kind"] = "no_coverage"
            q["note"] = (
                "Graded unanswerable by reading the retrieved passages: "
                f"{cand.get('grade_reason','')}. The cross-encoder scored it {score:.4f}, "
                "which illustrates why the score cannot be trusted as evidence of presence. "
                "Kept rather than deleted so the query set is not biased toward what the "
                "corpus happens to hold. See EXPERIMENT_LOG.md §6n."
            )
        elif tier == "marginal":
            q["note"] = (
                f"Partial support: {cand.get('grade_reason','')}. Counted as answerable, but "
                "a refusal here is defensible rather than a clear error. Cross-encoder "
                f"score {score:.4f}."
            )
        plan.append(q)

    print(f"\n{len(plan)} queries to add:")
    for q in plan:
        print(f"  {q['id']}  {q['top1_rerank_score']:7.4f}  {q['coverage_tier']:8}  "
              f"{q['domain']:16} {q['query'][:46]}")
    print(f"\n  strong {tally['strong']} | marginal {tally['marginal']} | none {tally['none']}")

    answerable_now = sum(1 for q in silver["queries"] if q.get("expect_answerable"))
    answerable_new = answerable_now + sum(1 for q in plan if q["expect_answerable"])
    print(f"\n  answerable queries: {answerable_now} -> {answerable_new}")
    print(f"  total queries:      {len(silver['queries'])} -> {len(silver['queries']) + len(plan)}")

    if not apply:
        print("\ndry run — nothing written. Re-run with --apply")
        return 0

    backup = SILVER.with_suffix(f".backup_{date.today().isoformat()}_pre_merge.json")
    shutil.copy2(SILVER, backup)
    silver["queries"].extend(plan)
    silver["_about"]["extension_2026_09_26"] = (
        f"Extended with {len(plan)} domain-derived queries covering pest damage, soil, seed "
        "treatment, season, water, planting and physical injury — the subject areas these "
        "treatises actually cover. The original 22 queries were almost entirely modern "
        "fungal leaf diseases, which §6m showed the texts cannot address, so the evaluation "
        "was measuring the system against unanswerable questions. Candidates were written "
        "from the domain WITHOUT consulting the corpus (to avoid circularity), then labelled "
        f"from measured cross-encoder scores: {tally['strong']} strong, {tally['marginal']} "
        f"marginal, {tally['none']} no-coverage. Queries scoring 'none' were kept as honest "
        "negatives, not deleted."
    )
    SILVER.write_text(json.dumps(silver, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {SILVER.relative_to(ROOT)}")
    print(f"backup {backup.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
