"""Generate notebooks/phase11_evaluation.ipynb.

Kept as a generator script (same pattern as the other phase notebooks) so the
notebook is reviewable as plain Python and regenerating it is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "phase11_evaluation.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


CELLS = [
    md("""\
# Phase 11 — Evaluation

Runs the query set through the **full system and its baselines**, and reports:

1. **Retrieval quality** — Precision@k, nDCG, MRR, Hit@k for the full system vs
   a keyword-only baseline and the stage ablations.
2. **Grounding** — do the answers cite passages that were *actually retrieved*?
   (deterministic citation verification — free, no judge model needed)
3. **Refusal behaviour** — honest refusal on the deliberately-unanswerable
   queries, and **over-refusal** on answerable ones.
4. **Ungrounded control** — the same LLM with no corpus, to show what grounding buys.
5. **RAGAS faithfulness / answer-relevancy** — judged by the **local Llama**, so
   there is no API cost.

> **These numbers are PRELIMINARY.** The query set is *silver* (project-authored).
> Labels are book-level, so **Recall@k is reported as `n/a`** — it is undefined
> until passage-level labels exist. When the expert gold-set arrives, drop it in
> and re-run: Recall switches on automatically and nothing else changes.

**Requirements:** T4 GPU runtime + HF token (private corpus + Llama).
"""),

    md("## Cell 1 — clone + dependencies"),
    code("""\
import os, shutil, subprocess, sys

REPO_PATH = "/content/iks-rag-thesis"
REPO_URL = "https://github.com/ankit8453/iks-rag-thesis.git"

os.chdir("/content")
shutil.rmtree(REPO_PATH, ignore_errors=True)
env = os.environ.copy()
env["GIT_LFS_SKIP_SMUDGE"] = "1"          # dodge the LFS bandwidth stall
r = subprocess.run(["git", "clone", REPO_URL, REPO_PATH], env=env,
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout); print(r.stderr)
    raise RuntimeError(f"git clone failed (exit {r.returncode})")
os.chdir(REPO_PATH); sys.path.insert(0, REPO_PATH)
print("Repo at:", os.getcwd())

DEPS = [
    "transformers>=4.44,<4.50", "accelerate>=0.34", "bitsandbytes>=0.44",
    "sentence-transformers>=3.0", "chromadb>=0.5", "rank_bm25>=0.2.2",
    "datasets>=2.20", "huggingface_hub>=0.24", "pyyaml>=6.0", "pydantic>=2.7",
]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS],
                   capture_output=True, text=True)
if r.returncode != 0:
    print("\\n".join(r.stdout.splitlines()[-30:]))
    raise RuntimeError("pip install failed")

# RAGAS is optional: if it will not install, the harness still reports every
# other metric and simply records that RAGAS was skipped.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ragas>=0.1.9"])
print("deps installed")

from huggingface_hub import login
login()

import torch
print("cuda:", torch.cuda.is_available())"""),

    md("## Cell 2 — build the corpus collection (CPU) and score retrieval\n\n"
       "Retrieval needs no generation, so this half runs before the LLM is loaded."),
    code("""\
from src.rag.corpus_loader import load_chunks_from_hf, build_chroma
from src.eval.query_set import load_query_set, answerable_cases
from src.eval.harness import run_full_evaluation

chunks = load_chunks_from_hf()
print("corpus chunks:", len(chunks))

collection = build_chroma(chunks, persist_dir="/content/eval_vecdb")
cases = load_query_set()
print(f"queries: {len(cases)} "
      f"({len(answerable_cases(cases))} answerable, "
      f"{len(cases) - len(answerable_cases(cases))} deliberate negatives)")

K = 5
retrieval_only = run_full_evaluation(cases, collection=collection, k=K)
print()
print(retrieval_only["retrieval_table"])"""),

    md("""\
### Reading the retrieval table

`full` is the system. **`keyword_only` is the baseline that matters** — if the
full system does not clearly beat it, the semantic bridge is not earning its
place. `dense_only` and `hybrid_no_rerank` isolate which stage contributes what.

`R@5` shows `n/a` by design: recall is undefined under book-level labels."""),

    md("## Cell 3 — load the LLM and run generation + the ungrounded control"),
    code("""\
from app.loaders import load_all

bundle = load_all()          # disease/soil engines + RAG pipeline + Llama
print("models ready")

full_eval = run_full_evaluation(
    cases, collection=collection,
    pipeline=bundle.rag_pipeline,     # grounded system
    llm=bundle.llm,                   # same model, used ungrounded as the control
    k=K,
)

g = full_eval["generation"]
print(f"answerable queries      : {g.n_answerable}")
print(f"grounded answer rate    : {g.grounded_answer_rate:.2%}   "
      f"(answers citing a genuinely retrieved passage)")
print(f"valid citation rate     : {g.valid_citation_rate:.2%}   "
      f"(citations pointing at real retrieved passages)")
print(f"honest refusal rate     : {g.honest_refusal_rate:.2%}   "
      f"(on the {g.n_negative} unanswerable queries - higher is better)")
print(f"over-refusal rate       : {g.over_refusal_rate:.2%}   "
      f"(refusing answerable queries - LOWER is better)")

u = full_eval["ungrounded"]
print(f"\\nungrounded control      : {u['n']} answers with no corpus at all; "
      f"unfounded citation rate {u['unfounded_citation_rate']:.2%}")"""),

    md("## Cell 4 — RAGAS faithfulness (judged by the local Llama, no API cost)"),
    code("""\
from src.eval.config import EvalConfig
from src.eval.ragas_eval import RAGEvalSample, run_ragas_evaluation

# Build one RAGAS row per answerable query from the grounded system's output.
samples = []
for case in answerable_cases(cases):
    res = bundle.rag_pipeline.answer(case.query, k=K)
    samples.append(RAGEvalSample(
        query=case.query,
        answer=getattr(res, "answer", "") or "",
        contexts=[c.text for c in getattr(res, "retrieved", [])],
        ground_truth=None,          # no expert reference answers yet
    ))
print("RAGAS samples:", len(samples))

cfg = EvalConfig()
cfg.ragas.metrics = ["faithfulness", "answer_relevancy"]   # the two that need no reference

# ---------------------------------------------------------------------------
# Judge selection. Leave False for the free run; flip to True for a second pass
# with an INDEPENDENT judge (see the note below for why that is worth doing).
# ---------------------------------------------------------------------------
USE_PAID_JUDGE = False

judge = judge_emb = None
if USE_PAID_JUDGE:
    import getpass, os, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langchain-openai"])
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")  # never hard-code
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # cheap, independent
    judge_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")  # local, free

scores = run_ragas_evaluation(
    samples, cfg,
    output_path="results/phase11_ragas_per_sample.csv",
    judge_llm=judge, judge_embeddings=judge_emb,
)
print("judge           :", "gpt-4o-mini (independent)" if USE_PAID_JUDGE else "none configured")
print("faithfulness    :", scores.faithfulness)
print("answer relevancy:", scores.answer_relevancy)
if scores.skipped:
    print("skipped:", scores.skipped)"""),

    md("""\
> **Cost guard.** With `USE_PAID_JUDGE = False` and no key set, RAGAS may simply
> report a missing key — that is fine, **skip this cell**. Cell 5 handles a skipped
> Cell 4 without failing. Citation verification in Cell 3 already measures grounding
> deterministically and for free, and is the claim to lead with.
>
> **Why a later paid pass is worth ~a few dollars.** Judging Llama's answers with
> Llama is self-evaluation — a reviewer can fairly object that the model marked its
> own homework. Re-running with `USE_PAID_JUDGE = True` (~22 queries, ~100 short
> calls) gives *"generated locally, judged independently"*, and reporting both
> judges side by side is stronger than either alone."""),

    md("## Cell 5 — save the results"),
    code("""\
import json, pathlib, datetime

# Cell 4 is optional — if it was skipped, carry on and record that RAGAS was
# not computed rather than failing the whole run at the last step.
try:
    scores
except NameError:
    from src.eval.ragas_eval import RAGASScores
    scores = RAGASScores(skipped={"_all": "Cell 4 skipped - no judge configured"})

out = {
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "PRELIMINARY - silver query set, book-level labels, "
              "pending the expert gold-set",
    "k": K,
    "n_answerable": full_eval["n_answerable"],
    "n_negative": full_eval["n_negative"],
    "retrieval": [r.as_row(K) for r in full_eval["retrieval"]],
    "generation": {
        "grounded_answer_rate": g.grounded_answer_rate,
        "valid_citation_rate": g.valid_citation_rate,
        "honest_refusal_rate": g.honest_refusal_rate,
        "over_refusal_rate": g.over_refusal_rate,
        "per_query": g.per_query,
    },
    "ungrounded_control": {k: v for k, v in u.items() if k != "answers"},
    "ragas": scores.as_row(),
    "ragas_skipped": scores.skipped,
    "ragas_judge": ("gpt-4o-mini (independent)"
                    if globals().get("USE_PAID_JUDGE") else "none / local"),
}
# Name the file after the judge so a later independent-judge pass ADDS a second
# data point instead of overwriting this one.
suffix = "paidjudge" if globals().get("USE_PAID_JUDGE") else "free"
p = pathlib.Path("results"); p.mkdir(exist_ok=True)
target = p / f"phase11_results_{suffix}.json"
target.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out["retrieval"], indent=2))
print(f"\\nSaved -> {target}")
print("Download it and share with Claude Code to write up the results.")"""),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
