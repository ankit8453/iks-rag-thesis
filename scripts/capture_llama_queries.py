"""Record what Strategy B (Llama) ACTUALLY writes, so it can be measured (§6o, Stage 0).

Every retrieval number in this project was measured on **hand-written** queries. The
deployed system does not use those: it feeds the disease label through Strategy B, and
Llama writes the query. Nobody has ever checked whether Llama's real output matches the
hand-written queries the evaluation assumes.

So the whole retrieval evaluation rests on an untested assumption. This script removes it:
it runs Strategy B over the disease labels in the silver set, saves the generated queries
in the same format the coverage checker reads, and lets both be scored the same way.

Three outcomes, all useful:
  * Llama scores higher  -> the hand-written queries were holding the system back; the
    evaluation should use Llama's real output.
  * Hand-written higher  -> Llama writes weak queries; fix the prompt (few-shot examples
    from the corpus, then a corpus word list, then two-round retrieval). Not fine-tuning.
  * Roughly equal        -> the bridge is sound and the limit really is corpus coverage.

The vision models are NOT run. The disease label is taken from the query set, because that
label is exactly what the vision model would hand to Strategy B. This isolates the
label -> query step, which is the only step under test.

**Run this on Colab** (needs Llama-3.1-8B, ~5 min to load, free T4 is enough). It writes
``data/eval/llama_generated_queries.json``; bring that file back and score it locally:

    python scripts/check_coverage.py data/eval/llama_generated_queries.json llama
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERY_SET = ROOT / "data" / "eval" / "silver_queries.json"
OUT = ROOT / "data" / "eval" / "llama_generated_queries.json"

#: Soil values to pair with each disease label. Strategy B takes a full multimodal
#: context, so something must be supplied. A neutral, unremarkable soil is used for every
#: query so that any difference between queries comes from the disease label alone.
NEUTRAL_SOIL = {
    "soil_type": "loam",
    "moisture_appearance": "moderately_moist",
    "texture": "medium",
}


def _context(disease: str, crop: str):
    """Build the MultimodalContext Strategy B expects, without running vision models."""
    from src.disease.model import DiseasePrediction
    from src.integration.causation import CausalContext, CausalPathway
    from src.integration.context import MultimodalContext
    from src.soil.model import SoilPrediction

    return MultimodalContext(
        disease_pred=DiseasePrediction(
            class_index=0, class_name=disease, confidence=0.85, logits=[],
        ),
        soil_pred=SoilPrediction(
            soil_type=NEUTRAL_SOIL["soil_type"],
            moisture_appearance=NEUTRAL_SOIL["moisture_appearance"],
            texture=NEUTRAL_SOIL["texture"],
            per_head_confidence={"soil_type": 0.9, "moisture": 0.9, "texture": 0.7},
        ),
        crop_type=crop or "general",
        # UNKNOWN pathway: no extra clause is injected, so the query reflects the label only.
        causal_context=CausalContext(pathway=CausalPathway.UNKNOWN, notes=None),
        user_notes=None,
    )


def main() -> int:
    from src.integration.strategy_llm_mediated import LLMMediatedStrategy
    from src.rag.generator import GroundedGenerator

    payload = json.loads(QUERY_SET.read_text(encoding="utf-8"))
    # Only queries that came from a vision label can be reproduced by the bridge; the
    # domain-style queries (added 2026-09-26) have no disease label and are not in scope.
    cases = [q for q in payload["queries"] if q.get("disease")]
    print(f"{len(cases)} disease-labelled queries to reproduce through Strategy B")

    print("loading Llama-3.1-8B (about 5 minutes) ...")
    llm = GroundedGenerator()
    if hasattr(llm, "load"):
        llm.load()
    strategy = LLMMediatedStrategy()

    out = []
    for i, q in enumerate(cases, 1):
        disease, crop = q["disease"], q.get("crop", "")
        try:
            generated = strategy.build_query(_context(disease, crop), llm)
        except Exception as exc:                      # noqa: BLE001 - record, never abort
            print(f"  [{i:2}/{len(cases)}] {q['id']}  FAILED: {exc}")
            generated, error = "", str(exc)
        else:
            error = None
            print(f"  [{i:2}/{len(cases)}] {q['id']}  {generated[:78]}")
        out.append({
            "id": q["id"],
            "crop": crop,
            "disease": disease,
            "query": generated,                       # what check_coverage.py will score
            "hand_written_query": q["query"],         # the side-by-side comparison
            "hand_written_top1": q.get("top1_rerank_score"),
            "hand_written_tier": q.get("coverage_tier"),
            "relevant_books": q.get("relevant_books", []),
            "relevant_chunk_ids": [],
            "error": error,
        })

    OUT.write_text(json.dumps({
        "_about": {
            "purpose": "Queries as ACTUALLY generated by Strategy B (Llama-3.1-8B) from the "
                       "disease labels in the silver set. Recorded so the bridge can be "
                       "measured instead of assumed -- every retrieval number so far was "
                       "computed on hand-written queries the deployed system never sends. "
                       "See EXPERIMENT_LOG.md 6o.",
            "method": "Vision models not run; the disease label is taken from the query set, "
                      "since that label is exactly what the vision model hands to Strategy B. "
                      "A neutral soil context is held constant across all queries so that any "
                      "difference comes from the disease label alone. Causal pathway UNKNOWN, "
                      "so no extra clause is injected.",
            "temperature": 0.2,
            "seed": 42,
            "soil_context": NEUTRAL_SOIL,
            "generated_on": "Colab, Llama-3.1-8B 4-bit",
        },
        "queries": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in out if r["query"])
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({ok}/{len(out)} generated)")
    print("\nNow score it locally (no GPU needed):")
    print(f"  python scripts/check_coverage.py {OUT.relative_to(ROOT)} llama")
    print("then compare against the hand-written scores:")
    print("  python scripts/compare_query_sources.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
