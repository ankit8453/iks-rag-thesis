"""Re-label the 16 domain queries' ``relevant_books`` independently of retrieval (§6n fix).

The bug being fixed. ``merge_new_queries.py`` set each new query's ``relevant_books`` to
``sorted({p["book"] for p in row["top5"][:3]})`` — the books retrieval had just returned.
Precision@5, nDCG and MRR then graded retrieval against its own output. That is marking
your own exam paper, and it inflated every retrieval variant; ``keyword_only`` jumping
from P@5 0.33 to 0.53 merely by adding queries was the visible symptom.

The original 24 queries are unaffected: their labels were authored by hand before any
retrieval ran.

The fix. Labels are assigned here from **what each treatise is about**, the same basis the
original 24 used — never from what came back in a ranking:

    vrikshayurveda        tree disorders, causes and remedies; planting; watering; soil;
                          seed treatment; wounds; unproductiveness
    brihat_samhita        rainfall prediction and omens; crop-growth prognostics;
                          exploration of water springs (ch.54); treatment of trees (ch.55)
    krishi_parashara      seasonal agronomy; sowing calendar; rain signs; ploughing
    upavanavinoda         arbori-horticulture; tree care; garden layout
    kashyapiyakrishisukti cultivation; land selection; soil and water management; seeds;
                          sowing; harvest
    niti_natural_farming  (modern) insect-pest management; bio-input recipes; seed
                          treatment; soil health; per-crop practice

Honesty note. This is not a blind labelling: some of these passages were read while
grading answerability. But the judgement here is "which treatise covers this subject",
which is answerable from each book's scope alone and was in several cases assigned against
what retrieval actually returned (n02's top-1 was a passage on raising plants from seed;
n13's was on where to site a vegetable plot). Author-assigned, pending expert ratification
like the rest of this silver set.

Usage:  python scripts/relabel_new_queries.py            # dry run
        python scripts/relabel_new_queries.py --apply
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "eval" / "silver_queries.json"

#: query id -> (books that should hold an answer, why — from the books' subject matter)
LABELS: dict[str, tuple[list[str], str]] = {
    "n01": (["niti_natural_farming", "vrikshayurveda"],
            "sucking-insect damage: NITI covers pest identification and sprays; "
            "Vrikshayurveda treats insect attack on trees"),
    "n02": (["niti_natural_farming", "vrikshayurveda"],
            "chewing-insect damage: same two sources cover insect attack"),
    "n03": (["vrikshayurveda", "upavanavinoda"],
            "ants are named explicitly among Vrikshayurveda's tree disorders; "
            "Upavanavinoda covers the same arbori-horticultural ground"),
    "n04": (["niti_natural_farming", "vrikshayurveda"],
            "preparation of a pest deterrent: NITI's bio-input formulations; "
            "Vrikshayurveda's fumigation and decoction remedies"),
    "n05": (["kashyapiyakrishisukti", "niti_natural_farming", "vrikshayurveda"],
            "seed treatment before sowing is covered by all three: Kashyapiya on seeds "
            "and sowing, NITI's Beejamrit, Vrikshayurveda's seed-treatment verses"),
    "n06": (["vrikshayurveda", "kashyapiyakrishisukti", "upavanavinoda"],
            "land suitability for planting: Vrikshayurveda describes land recommended "
            "for planting, Kashyapiya land selection, Upavanavinoda garden siting"),
    "n07": (["kashyapiyakrishisukti", "vrikshayurveda", "niti_natural_farming"],
            "soil enrichment before planting: Kashyapiya ploughing and cow dung, "
            "Vrikshayurveda manuring, NITI soil-health management"),
    "n08": (["vrikshayurveda", "upavanavinoda"],
            "kunapajala, the classical liquid manure, is a Vrikshayurveda subject and "
            "recurs in Upavanavinoda"),
    "n09": (["krishi_parashara", "kashyapiyakrishisukti"],
            "the sowing calendar is core Krishi Parashara; Kashyapiya covers seasonal "
            "cultivation"),
    "n10": (["brihat_samhita", "krishi_parashara"],
            "rainfall prognostication from natural signs is the subject of Brihat "
            "Samhita ch.21-28 and of Krishi Parashara's rain chapters"),
    "n11": (["vrikshayurveda", "upavanavinoda"],
            "watering schedules for young trees are a Vrikshayurveda subject; "
            "Upavanavinoda covers the same care regime"),
    "n12": (["brihat_samhita", "kashyapiyakrishisukti"],
            "Brihat Samhita ch.54 is 'Exploration of Water Springs'; Kashyapiya covers "
            "well digging and water management"),
    "n13": (["vrikshayurveda", "upavanavinoda", "kashyapiyakrishisukti"],
            "planting and transplanting method: Vrikshayurveda and Upavanavinoda on "
            "saplings, Kashyapiya on cultivation practice"),
    "n14": (["vrikshayurveda", "upavanavinoda"],
            "physical wounds and broken branches are treated explicitly in "
            "Vrikshayurveda; Brihat ch.55 and Upavanavinoda cover tree treatment"),
    "n15": (["vrikshayurveda", "upavanavinoda"],
            "unproductiveness — trees that do not flower or fruit — is a named disorder "
            "in both"),
    "n16": (["kashyapiyakrishisukti", "krishi_parashara"],
            "post-harvest handling and storage would belong to the two cultivation "
            "treatises; graded 'no' because neither in fact addresses it"),
}


def main() -> int:
    apply = "--apply" in sys.argv
    payload = json.loads(SILVER.read_text(encoding="utf-8"))

    print(f"{'id':5} {'from retrieval (circular)':46} -> from the books' subject matter")
    changed = 0
    for q in payload["queries"]:
        entry = LABELS.get(q["id"])
        if entry is None or q.get("added") != "2026-09-26":
            continue
        books, why = entry
        before = list(q.get("relevant_books") or [])
        if before != books:
            changed += 1
        print(f"{q['id']:5} {', '.join(before)[:46]:46} -> {', '.join(books)}")
        if apply:
            q["relevant_books"] = books
            q["relevant_books_basis"] = why

    print(f"\n{changed} of {len(LABELS)} labels change")
    if not apply:
        print("dry run - re-run with --apply")
        return 0

    backup = SILVER.with_suffix(f".backup_{date.today().isoformat()}_pre_relabel.json")
    shutil.copy2(SILVER, backup)
    payload["_about"]["relevant_books_relabel_2026_09_26"] = (
        "The 16 domain queries added on 2026-09-26 initially took relevant_books from the "
        "top-3 books of their own retrieval run, which made Precision@5, nDCG and MRR "
        "circular - retrieval graded against its own output. Symptom: keyword_only P@5 rose "
        "0.33 -> 0.53 purely from adding queries. They are now labelled from each "
        "treatise's subject matter, the same basis as the original 24, with the reason "
        "recorded per query in relevant_books_basis. Author-assigned, pending expert "
        "ratification. Generation metrics were never affected: they use expect_answerable "
        "and citation resolution, never relevant_books.")
    SILVER.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {SILVER.relative_to(ROOT)}")
    print(f"backup {backup.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
