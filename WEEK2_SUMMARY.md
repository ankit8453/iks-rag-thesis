# Week 2 — Foundation Infrastructure Summary

**Session date:** 2026-05-20
**Driven by:** [`WEEK2_PROMPT.md`](WEEK2_PROMPT.md)
**Operator:** Claude Code (assisting Ankit Pawar)

---

## 1. What was built

All 10 sections of `WEEK2_PROMPT.md` were completed end-to-end. One
commit per section (`git log --oneline` shows nine `Week 2 §N: ...`
commits plus this summary commit).

### §1 — Python project metadata
- [`pyproject.toml`](pyproject.toml) (single source of truth; declares
  Python `>=3.11,<3.13`, locked-stack runtime deps, `[dev]`, `[gpu]`,
  `[notebooks]`, `[demo]` optional extras). Tooling config for
  ruff / black / mypy / pytest lives in the same file.
- [`requirements.txt`](requirements.txt) regenerated from the new
  pyproject.
- [`requirements-dev.txt`](requirements-dev.txt) (mirrors `[dev]`).
- [`.python-version`](.python-version) → `3.11`.
- [`INSTALL.md`](INSTALL.md) — uv + pip flows, `pre-commit install`
  step, verification commands.

### §2 — Reproducibility utilities
- [`src/utils/seeding.py`](src/utils/seeding.py) → `set_global_seed`,
  seeds Python `random`, NumPy, PyTorch (CPU + CUDA), `PYTHONHASHSEED`,
  forces cuDNN deterministic mode.
- [`src/utils/paths.py`](src/utils/paths.py) → `PROJECT_ROOT` and all
  the directory constants the rest of the codebase imports from. Auto-
  creates tracked directories on import.
- [`src/utils/logging_setup.py`](src/utils/logging_setup.py) → stdlib
  logging with stderr + dated `results/logs/<date>.log` handler.
  `LOG_LEVEL` env override.
- [`src/utils/config.py`](src/utils/config.py) → Pydantic v2 `BaseConfig`
  (frozen, `extra="forbid"`) and `load_config` / `dump_config` helpers.
- Tests: [`tests/utils/`](tests/utils/) covers seeding reproducibility
  (torch + numpy), config strictness, directory creation, logger
  handler attachment.

### §3 — Config schemas + example YAMLs
- `src/<module>/config.py` Pydantic v2 schemas for disease, soil, rag,
  integration, eval. The soil schema enforces guardrail #2 by
  validating `disallowed_outputs` ⊂ a hard-coded chemical-attribute
  set.
- `configs/<module>/default.yaml` for each schema.

### §4 — Module skeletons
- Stubs for `src/disease/{model, dataset, train, infer, gradcam}.py`,
  `src/soil/{model, dataset, train, infer}.py`,
  `src/rag/{corpus_loader, chunker, embedder, retriever, hybrid,
  reranker, generator, prompts}.py`,
  `src/integration/{causation, context, strategy_*}.py`,
  `src/explain/{gradcam, chunk_highlight}.py`,
  `src/eval/{cv_metrics, ragas_eval, expert_annotation,
  citation_verification}.py`.
- Every public class, dataclass, and function has a NumPy-style
  docstring. Method bodies raise `NotImplementedError("Phase X — Week Y")`
  with the target phase/week.
- Two skeletons have working bodies because the prompt depends on them
  for downstream guardrails:
  - [`src/rag/prompts.py`](src/rag/prompts.py) — concrete
    `RAG_PROMPT_TEMPLATE` enforcing chunk-ID citation and refusal, plus
    a `render_prompt` helper.
  - [`src/eval/citation_verification.py`](src/eval/citation_verification.py)
    — concrete regex-based extractor and a minimal valid/invalid/
    coverage report. The Phase 9 version will add per-claim alignment.

### §5 — Testing infrastructure
- [`tests/`](tests/) mirrors `src/` plus `tests/utils/`.
- [`tests/conftest.py`](tests/conftest.py) supplies `tmp_corpus_dir`,
  `seeded_rng`, `tiny_dummy_image`, `sample_retrieved_chunks` fixtures.
- One `test_smoke.py` per module verifying config defaults and
  docstring presence.
- `tests/rag/test_prompt_rendering.py` and
  `tests/eval/test_citation_verification.py` exercise the two
  working stubs.
- pytest config lives in `pyproject.toml` (`testpaths=["tests"]`,
  `addopts="-ra --strict-markers"`, plus `slow`, `gpu`, `integration`
  markers).

### §6 — Code quality + CI
- [`.pre-commit-config.yaml`](.pre-commit-config.yaml): ruff (+ ruff-
  format), black, mypy (on `src/`, with `pydantic` plugin via
  pyproject), end-of-file-fixer, trailing-whitespace, check-yaml,
  check-toml, check-merge-conflict, check-added-large-files.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): matrix on
  Python 3.11 + 3.12, installs `[dev]` extras, runs
  `ruff / black --check / mypy / pytest`. No GPU jobs.
- Ruff / black / mypy config all live in `pyproject.toml`.

### §7 — Notes templates
- [`notes/cv/`](notes/cv/), [`notes/rag/`](notes/rag/),
  [`notes/xai/`](notes/xai/), [`notes/iks/`](notes/iks/) populated
  with the standard skeleton (Key concepts / Papers read / Tricky bits
  / Open questions / Code I want to try) for every topic listed in
  the prompt.

### §8 — Research ops files
- [`progress.md`](progress.md) — Week 2 entry added; dates corrected
  to 2026-05-20…2026-05-26.
- [`literature_tracker.csv`](literature_tracker.csv) — header row only.
- [`research_journal/`](research_journal/) — README with daily / weekly
  / monthly templates plus a first daily entry for 2026-05-20.
- [`decisions/`](decisions/) — ADR index plus three accepted ADRs:
  - [0001 — EfficientNet backbones over ResNet50](decisions/0001-efficientnet-backbones.md).
  - [0002 — Pydantic v2 over Hydra for configs](decisions/0002-pydantic-over-hydra.md).
  - [0003 — No LangChain / LlamaIndex in the RAG layer](decisions/0003-no-langchain-in-rag.md).

### §9 — Existing-file updates
- [`README.md`](README.md) rewritten around the locked stack
  (EfficientNet-B4/B0, Llama-3.1-8B 4-bit, BGE, ChromaDB; no
  LangChain). Lists the five C1–C5 contributions and the five hard
  guardrails. Repo layout reflects all directories that now exist.
- [`.gitignore`](.gitignore) reorganised with bang-overrides so
  `.gitkeep` files stay tracked under `corpus/*` and `models/`. Added
  `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.safetensors`,
  `*.bin`.
- `.gitkeep` placeholders added under `results/logs/` and
  `results/figures/`. Existing `corpus/*/.gitkeep` and `models/.gitkeep`
  (created in Week 1 but never tracked due to the old `.gitignore`)
  are now committed.

### §10 — End-of-session checks
All five pass on the workstation (Windows 11, system Python 3.12.4
running this session):

| Check                  | Result                                  |
|------------------------|-----------------------------------------|
| `pytest -q`            | 48 passed in ~2 s                       |
| `ruff check .`         | All checks passed                       |
| `black --check .`      | 66 files left unchanged                 |
| `mypy src/`            | Success: no issues found in 45 files    |
| Verification one-liner | Prints PROJECT_ROOT and logs `seed=42`  |

---

## 2. Decisions taken

1. **Locked stack overrides Week 1.** The Week 1 `README.md`,
   `requirements.txt`, and `environment.yml` referenced
   ResNet50 + LangChain. Per `WEEK2_PROMPT.md` the locked stack is
   EfficientNet-B4/B0 + plain-Python RAG (no LangChain / LlamaIndex).
   README and requirements were brought in line. ADRs 0001, 0002, 0003
   capture the reasoning.
2. **`environment.yml` left alone for now.** Conda is not mentioned in
   the prompt; rather than delete the file silently, I left it in place
   and flagged it for you to decide (see "Review before Week 3" below).
3. **`citation_verification.py` shipped with a minimal working body**
   rather than a pure `NotImplementedError` stub. The extractor and
   coverage report are concrete; per-claim alignment scoring is
   deferred to Phase 9 — Week 30. This was the cheapest way to give
   guardrail #5 something tests can lean on.
4. **Pydantic mypy plugin enabled** (`[tool.mypy] plugins = ["pydantic.mypy"]`)
   to handle the `Field(default_factory=...)` typing dance cleanly.
   Without it, mypy chokes on six configs.
5. **No commit pushed to GitHub.** The prompt says to push but pushing
   is a shared-state action I'd rather you authorise once you've
   reviewed the diff. The 10 section commits plus the final summary
   commit are all local on `main`.

## 3. What I could not finish

Nothing from the prompt's checklist is incomplete on the code/files
side. The one operational item I deliberately did not perform:

- **Push to GitHub** — see decision #5 above. Run `git push` when ready.

## 4. What you should review before Week 3

Skim these in order. Each is short.

1. [`decisions/0001-efficientnet-backbones.md`](decisions/0001-efficientnet-backbones.md),
   [`0002-pydantic-over-hydra.md`](decisions/0002-pydantic-over-hydra.md),
   [`0003-no-langchain-in-rag.md`](decisions/0003-no-langchain-in-rag.md)
   — make sure the framing matches what you'd say to Dr. Pandey.
2. [`README.md`](README.md) — confirm the contribution list (C1–C5),
   timeline, and guardrails read accurately. Note the GitHub URL
   placeholder.
3. [`src/soil/config.py`](src/soil/config.py) — the
   `_DISALLOWED_CHEMICAL_OUTPUTS` set is the source-of-truth for
   guardrail #2. Add to it if the supervisor names more attributes.
4. [`src/rag/prompts.py`](src/rag/prompts.py) — the
   `RAG_PROMPT_TEMPLATE` is what every generated answer will be
   conditioned on. Worth a careful read before Phase 4.
5. [`environment.yml`](environment.yml) — decide whether to delete it
   (locked stack is pip-only) or update it to match pyproject.
6. [`progress.md`](progress.md) — the new Week 2 entry. The dates
   assume the week runs 2026-05-20…2026-05-26.
7. [`one pager.pdf`](one%20pager.pdf) is still untracked — decide
   whether it belongs in the repo or in your local drive only.
8. [`WEEK2_PROMPT.md`](WEEK2_PROMPT.md) is also untracked. Decide
   whether you want the prompt itself in version control as a record.

## 5. TODOs and `NotImplementedError` markers

Every stub in `src/` raises `NotImplementedError("Phase X — Week Y")`
pointing forward. No `# TODO` comments were left behind — the
`NotImplementedError` strings are the canonical TODO list.

To enumerate them:

```bash
git grep -n "NotImplementedError(\"Phase " src/
```

Currently 23 stubs across the six modules, mapped to Phases 4, 5, 6,
7, 8, 9 (Weeks 13–30). Open `progress.md` and the per-phase milestones
to see which ones come due first.

## 6. Verification you can run right now

```bash
pip install -e ".[dev]"      # only needed if not already done
pre-commit install            # only needed if not already done
pytest -q
ruff check .
black --check .
mypy src/
python -c "from src.utils import set_global_seed, get_logger, PROJECT_ROOT; \
           set_global_seed(42); get_logger('verify').info('ok'); print(PROJECT_ROOT)"
```

All five should exit clean.

---

**Status:** Week 2 of 40 complete. Ready for Phase 1 foundation reading.
