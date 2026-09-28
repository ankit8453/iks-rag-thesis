"""Phase 11 harness — runs the query set through the system and its baselines.

Produces the comparison table the thesis needs:

* **Retrieval** — Precision@k / nDCG / MRR / Hit@k for the full system against
  the keyword-only baseline and the stage ablations (Recall@k appears once the
  query set carries passage-level labels).
* **Generation** — for the answerable queries, whether the grounded answer cites
  real retrieved passages (via citation verification, which is free and
  deterministic); for the deliberate negatives, whether the system honestly
  refuses.
* **Ungrounded control** — the same LLM with no corpus, to show what grounding
  is actually buying.

Everything model-shaped is injected, so this module holds no model, loads
nothing, and is unit-testable with fakes. The expensive pass runs on Colab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.eval.baselines import (
    RETRIEVAL_VARIANTS,
    RetrievalVariant,
    answer_without_grounding,
    chunks_to_pairs,
    PARTIAL,
    REFUSED,
    classify_answer,
    is_refusal,
)
from src.eval.citation_verification import verify_citations_in_context
from src.eval.query_set import QueryCase, aggregate, score_case
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class RetrievalRun:
    """Retrieval scores for one variant over the query set."""

    variant: str
    description: str
    is_baseline: bool
    summary: dict[str, Any]
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def as_row(self, k: int) -> dict[str, Any]:
        s = self.summary
        recall = s.get("recall")
        return {
            "variant": self.variant,
            "baseline": self.is_baseline,
            "n": s.get("n", 0),
            f"P@{k}": _r(s.get("precision")),
            # "n/a", not a number and not an em-dash: recall is undefined under
            # book-level labels, and the table must print on a cp1252 console.
            f"R@{k}": _r(recall) if recall is not None else "n/a",
            f"nDCG@{k}": _r(s.get("ndcg")),
            "MRR": _r(s.get("mrr")),
            f"Hit@{k}": _r(s.get("hit")),
        }


@dataclass
class GenerationRun:
    """Grounding quality over the answerable queries + refusal behaviour."""

    n_answerable: int = 0
    n_negative: int = 0
    #: fraction of answers citing at least one genuinely retrieved passage
    grounded_answer_rate: float = 0.0
    #: fraction of citations that pointed at a passage actually retrieved
    valid_citation_rate: float = 0.0
    #: on the unanswerable queries, how often it did NOT give a full answer
    #: (refused, or answered only in part with a caveat)
    honest_refusal_rate: float = 0.0
    #: on answerable queries, how often it refused outright - declined and cited nothing.
    #: A cited answer with a trailing caveat is "partial", not a refusal (6p).
    over_refusal_rate: float = 0.0
    #: on answerable queries, how often it answered part and declined part
    partial_rate: float = 0.0
    #: outcome counts, so the three classes can be reported rather than inferred
    outcomes_answerable: dict[str, int] = field(default_factory=dict)
    outcomes_unanswerable: dict[str, int] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)


def _r(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, (int, float)) else value


def run_retrieval_eval(
    cases: list[QueryCase],
    retriever: Any,
    variant: RetrievalVariant,
    k: int = 5,
) -> RetrievalRun:
    """Score one retrieval variant across the answerable queries."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        chunks = retriever.retrieve(case.query, k=k)
        row = score_case(case, chunks_to_pairs(chunks), k=k)
        rows.append(row)
    return RetrievalRun(
        variant=variant.name, description=variant.description,
        is_baseline=variant.is_baseline,
        summary=aggregate(rows, k=k), per_query=rows,
    )


def run_generation_eval(
    cases: list[QueryCase],
    pipeline: Any,
    k: int = 5,
    *,
    progress: bool = True,
    save_path: str | None = None,
) -> GenerationRun:
    """Measure grounding + refusal behaviour of the full system.

    Grounding is judged by citation verification — deterministic, free, and it
    directly answers "is this recommendation traceable to a real passage?",
    which is the project's central claim.
    """
    import json as _json  # noqa: PLC0415
    import time as _time  # noqa: PLC0415

    run = GenerationRun()
    grounded_hits, citation_rates, refusals_neg, refusals_pos, partial_pos = [], [], [], [], []
    out_ans: dict[str, int] = {}
    out_neg: dict[str, int] = {}
    # Each answer is appended to disk the moment it exists, so a Colab disconnect late in
    # a ~90-minute run loses nothing already generated.
    sink = open(save_path, "a", encoding="utf-8") if save_path else None  # noqa: SIM115
    t0 = _time.time()

    for n, case in enumerate(cases, 1):
        result = pipeline.answer(case.query, k=k)
        answer = getattr(result, "answer", "") or ""
        retrieved_ids = [c.chunk_id for c in getattr(result, "retrieved", [])]
        # The generator cites as "[Source, ch.X, v.Y]" and RESOLVES those to
        # chunk_ids itself (RAGAnswer.citations / .used_chunk_ids). Prefer that
        # over re-parsing the text: the chunk-id regex in citation_verification
        # cannot match the source/chapter/verse form and would score every
        # answer as ungrounded. Fall back to it only for pipelines that do not
        # expose the resolved fields.
        cited = list(getattr(result, "citations", []) or [])
        used_ids = list(getattr(result, "used_chunk_ids", []) or [])
        if not cited and not used_ids:
            report = verify_citations_in_context(answer, retrieved_ids)
            cited, used_ids = report.cited_ids, report.valid_ids
        # a citation is "valid" when it resolved to a genuinely retrieved chunk
        valid_ids = [cid for cid in used_ids if cid in set(retrieved_ids)]

        outcome = classify_answer(answer, len(valid_ids))
        refused = outcome == REFUSED

        if case.expect_answerable:
            run.n_answerable += 1
            out_ans[outcome] = out_ans.get(outcome, 0) + 1
            grounded_hits.append(1.0 if valid_ids else 0.0)
            if cited:
                citation_rates.append(min(1.0, len(valid_ids) / len(cited)))
            refusals_pos.append(1.0 if refused else 0.0)
            partial_pos.append(1.0 if outcome == PARTIAL else 0.0)
        else:
            run.n_negative += 1
            out_neg[outcome] = out_neg.get(outcome, 0) + 1
            # on an unanswerable query, declining the core question is the right call
            # whether it cites nothing or cites a general passage with a caveat
            refusals_neg.append(1.0 if outcome in (REFUSED, PARTIAL) else 0.0)

        row = {
            "id": case.id, "expect_answerable": case.expect_answerable,
            "outcome": outcome, "refused": refused,
            "n_cited": len(cited), "n_valid": len(valid_ids),
            "n_invalid": max(0, len(cited) - len(valid_ids)),
            "query": case.query,
            "retrieved_ids": retrieved_ids,
            # full text: 6p showed a truncated preview can hide the part that matters
            "answer": answer,
        }
        run.per_query.append(row)
        if sink is not None:
            sink.write(_json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
        if progress:
            el = _time.time() - t0
            eta = el / n * (len(cases) - n)
            print(f"  [{n:2}/{len(cases)}] {case.id:5} {outcome:8} "
                  f"valid {len(valid_ids)}/{len(cited)}  "
                  f"elapsed {el/60:5.1f} min  eta {eta/60:5.1f} min", flush=True)

    if sink is not None:
        sink.close()
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    run.grounded_answer_rate = mean(grounded_hits)
    run.valid_citation_rate = mean(citation_rates)
    run.honest_refusal_rate = mean(refusals_neg)
    run.over_refusal_rate = mean(refusals_pos)
    run.partial_rate = mean(partial_pos)
    run.outcomes_answerable = out_ans
    run.outcomes_unanswerable = out_neg
    return run


def run_ungrounded_control(
    cases: list[QueryCase], llm: Any, *, progress: bool = True,
) -> dict[str, Any]:
    """The 'do we need the corpus?' control — same LLM, no retrieved passages."""
    answers = []
    for n, c in enumerate(cases, 1):
        answers.append(answer_without_grounding(llm, c.query))
        if progress:
            print(f"  [control {n:2}/{len(cases)}] {c.id}", flush=True)
    cited = sum(1 for a in answers if verify_citations_in_context(a, []).cited_ids)
    return {
        "n": len(answers),
        # it has no retrieved passages, so any citation-looking text is unfounded
        "unfounded_citation_rate": (cited / len(answers)) if answers else 0.0,
        "answers": answers,
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    """Render result rows as a fixed-width table for the notebook / thesis."""
    if not rows:
        return "(no results)"
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    sep = "  ".join("-" * widths[h] for h in headers)
    body = "\n".join(
        "  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) for r in rows
    )
    return f"{line}\n{sep}\n{body}"


def run_full_evaluation(
    cases: list[QueryCase],
    *,
    collection: Any,
    pipeline: Any = None,
    llm: Any = None,
    k: int = 5,
    variants: tuple[RetrievalVariant, ...] = RETRIEVAL_VARIANTS,
    shared_models: dict[str, Any] | None = None,
    run_retrieval: bool = True,
    save_path: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run every retrieval variant, plus generation + the ungrounded control.

    ``pipeline`` and ``llm`` are optional so the retrieval half can be run on
    its own (it needs no GPU generation).

    ``run_retrieval=False`` skips the retrieval variants. The Phase 11 notebook scores
    retrieval in its own cell first; repeating all four variants inside the generation cell
    cost ~10 minutes of CPU per run for identical numbers.

    ``save_path`` appends each generated answer to a JSONL file as soon as it exists.
    """
    from src.eval.baselines import build_retriever  # noqa: PLC0415

    from src.eval.query_set import answerable_cases  # noqa: PLC0415

    answerable = answerable_cases(cases)
    negatives = [c for c in cases if not c.expect_answerable]

    retrieval: list[RetrievalRun] = []
    for variant in (variants if run_retrieval else ()):
        try:
            retriever = build_retriever(collection, variant, **(shared_models or {}))
            retrieval.append(run_retrieval_eval(answerable, retriever, variant, k=k))
        except Exception as exc:  # noqa: BLE001 - one variant must not kill the run
            _LOGGER.warning("Retrieval variant %s failed: %s", variant.name, exc)

    out: dict[str, Any] = {
        "k": k,
        "n_answerable": len(answerable),
        "n_negative": len(negatives),
        "retrieval": retrieval,
        "retrieval_table": format_table([r.as_row(k) for r in retrieval]),
    }
    if pipeline is not None:
        out["generation"] = run_generation_eval(
            cases, pipeline, k=k, progress=progress, save_path=save_path)
    if llm is not None:
        out["ungrounded"] = run_ungrounded_control(answerable, llm, progress=progress)
    return out


__all__ = [
    "GenerationRun",
    "RetrievalRun",
    "format_table",
    "run_full_evaluation",
    "run_generation_eval",
    "run_retrieval_eval",
    "run_ungrounded_control",
]
