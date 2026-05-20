# Cleanup Plan — Dry-Run Preview

> Branch: `cleanup/pdf-alignment` (Phase 0 done).
> No changes have been executed yet. After this file is committed, the
> session pauses for human review per the prompt's working-style note.

Tag legend: `[PDF §X]` direct PDF requirement · `[PDF-implied]`
required to satisfy a guardrail · `[ADDED]` engineering hygiene that
will be removed · `stale` superseded artifact.

---

## Will MOVE (Phase 2)

- `paper/thesis/` → `thesis/` (top-level)
  — `[PDF §41]` shows `paper/` and `thesis/` as **two separate** top-
  level directories, currently nested. Will use `git mv` so history is
  preserved. The lone `paper/thesis/.gitkeep` becomes `thesis/.gitkeep`,
  which is then replaced by `thesis/README.md` in Phase 4.5.

After the move, `paper/` will still hold its existing `paper/.gitkeep`
until Phase 4.5 adds `paper/README.md`.

## §41 directory-skeleton verification (Phase 2.2)

I checked the §41 tree against the current repo. Status:

| §41 directory      | Present? | Notes |
|--------------------|----------|-------|
| `data/{plant_disease, soil, splits}` | ✓ | `.gitkeep` in each |
| `corpus/{raw, cleaned, chunks, vector_db}` | ✓ | `.gitkeep` in each |
| `src/{disease, soil, rag, integration, explain, eval, utils}` | ✓ | populated stubs |
| `notebooks/` | ✓ on disk, **empty** | gets `README.md` + `00_environment_check.ipynb` in Phase 4 |
| `scripts/` | ✓ on disk, **empty** | gets `README.md` in Phase 4 |
| `configs/` | ✓ | five `default.yaml` files |
| `results/` | ✓ | tracked `.gitkeep`; `logs/` + `figures/` subdirs |
| `models/` | ✓ | `.gitkeep` |
| `demo/` | ✓ on disk, **empty** | gets `README.md` in Phase 4 |
| `paper/` | ✓ (currently has nested `thesis/`) | unnested in Phase 2.1 |
| `thesis/` | ✗ (currently inside `paper/`) | created by Phase 2.1 move |

Directories NOT in §41 that currently exist (and what will happen):

| Directory | Verdict |
|-----------|---------|
| `tests/` | **Keep** — explicitly outside §41 but `[PDF-implied §37]` for the reproducibility-test guardrail. |
| `notes/` | **Keep** — `[PDF-implied]` Phase-1 study scaffolding; not in the prompt's deletion list. |
| `research_journal/` | **Keep** — `[PDF §44]`. |
| `decisions/` | **Delete (Phase 3.3)** — `[ADDED]`. |
| `.github/` | **Delete (Phase 3.1)** — `[ADDED]` (CI). |

## Will DELETE (Phase 3)

### 3.1 / 3.2 / 3.3 — engineering hygiene + ADRs `[ADDED]`

| Path                                        | Reason |
|---------------------------------------------|--------|
| `.pre-commit-config.yaml`                   | `[ADDED]` pre-commit hooks |
| `.github/workflows/ci.yml`                  | `[ADDED]` GitHub Actions |
| `.github/workflows/` (becomes empty)        | parent dir cleanup |
| `.github/` (becomes empty)                  | parent dir cleanup |
| `pyproject.toml`                            | `[ADDED]` — §41 names `requirements.txt` + `environment.yml` only. Removing also deletes the ruff/black/mypy/pytest/pydantic-mypy config blocks. |
| `requirements-dev.txt`                      | `[ADDED]` — only `pytest` and `pydantic` survive the §22 trim and they land in `requirements.txt` (see Phase 5.1) |
| `.python-version`                           | `[ADDED]` pyenv pin |
| `decisions/0001-efficientnet-backbones.md`  | `[ADDED]` ADR |
| `decisions/0002-pydantic-over-hydra.md`     | `[ADDED]` ADR |
| `decisions/0003-no-langchain-in-rag.md`     | `[ADDED]` ADR (note: this ADR forbade LangChain; new `requirements.txt` per the prompt re-includes `langchain` as scaffolding-only — see Phase 5.1) |
| `decisions/README.md`                       | `[ADDED]` ADR index |
| `decisions/` (becomes empty)                | parent dir cleanup |

No standalone tool config files (`ruff.toml`, `.flake8`, `mypy.ini`, etc.) exist — verified with `ls`.

### 3.4 — session reports `[ADDED]`

| Path                | Reason |
|---------------------|--------|
| `WEEK2_SUMMARY.md`  | session report |
| `WEEK2_AUDIT.md`    | session report (its decisions are folded into this plan) |
| `WEEK2_PROMPT.md`   | session prompt (currently **untracked**, will be removed via `rm`) |
| `CLEANUP_PROMPT.md` | session prompt for THIS run (currently **untracked**) — will be removed via `rm` in Phase 3 alongside the others, matching the spirit of 3.4 |
| `CLEANUP_PLAN.md`   | **kept through Phases 2–5**; deleted in Phase 6 after verification |

### 3.5 — setup convenience `[ADDED]`

| Path         | Reason |
|--------------|--------|
| `INSTALL.md` | `[ADDED]` — essentials fold into `README.md` "Quick start" in Phase 5.4 |

### 3.6 — stale artifacts

| Path             | Reason |
|------------------|--------|
| `one pager.pdf`  | stale (ResNet50 + MiniLM-L6); currently **untracked**, will be removed via `rm` |

### 3.7 — what will NOT be deleted

Explicitly leaving alone (per prompt's "must NOT delete" list):

- All of `src/` (including the `SoilConfig.disallowed_outputs` validator, `RAG_PROMPT_TEMPLATE`, citation verification, `CausalPathway` enum, `cv_metrics.ClassificationReport`, `seeding`, etc.)
- All of `tests/`
- All of `configs/`
- `requirements.txt`, `environment.yml` (rewritten in Phase 5, not deleted)
- `README.md`, `progress.md`, `.gitignore` (rewritten in Phase 5)
- `literature_tracker.csv`, all of `research_journal/`
- All of `corpus/` and `data/` (.gitkeep files + structure)
- All of `notes/`

## Will CREATE (Phase 4)

### 4.1 — `BACKUP.md` `[PDF §43]`
Three-section 3-2-1 backup strategy (3 copies / 2 media / 1 off-site) plus an automation section. Exact text per the prompt.

### 4.2 — `references.bib` `[PDF §42]`
Empty BibTeX file with a single header comment pointing at Zotero / BetterBibTeX export.

### 4.3 — research journal templates `[PDF §44]`

| Path                                      | Contents |
|-------------------------------------------|----------|
| `research_journal/weekly/README.md`       | one-paragraph instructions for weekly entries |
| `research_journal/weekly/2026-W21.md`     | starter entry for the current week (today is 2026-05-21, week W21) with the four section headers |
| `research_journal/monthly/README.md`      | one-paragraph instructions for monthly entries |
| `research_journal/monthly/2026-05.md`     | starter entry for May 2026 with the standard headers |

### 4.4 — `notebooks/00_environment_check.ipynb` `[PDF-implied]`
Minimal six-cell Jupyter notebook (proper JSON, no executed outputs) verifying torch, timm, sentence-transformers, chromadb, `src.utils.seeding`, and `src.utils.paths`.

### 4.5 — empty-directory READMEs `[PDF-implied §41]`

| Path                  | Rationale (one-line each) |
|-----------------------|---------------------------|
| `scripts/README.md`   | placeholder noting `train_disease.py` / `train_soil.py` / `run_rag_eval.py` will land in Phases 4/5/7 |
| `demo/README.md`      | placeholder for the Streamlit app (Phase 9) |
| `paper/README.md`     | placeholder for LaTeX conference paper (Phase 13) |
| `thesis/README.md`    | placeholder for LaTeX M.Tech thesis (Phase 14) |
| `notebooks/README.md` | placeholder noting `00_environment_check.ipynb` is the first |

### 4.6 — `pytest.ini` `[PDF-implied §37]`
Minimal config (`testpaths = tests`, `addopts = -ra --strict-markers`) since pyproject is gone. The prompt's Phase 5.6 puts this in the "rewrite" phase, but it's a CREATE here because no `pytest.ini` currently exists.

## Will REWRITE (Phase 5)

| Path                | One-line summary of new contents |
|---------------------|----------------------------------|
| `requirements.txt`  | strict §22 alignment with category comments; **re-introduces `langchain` per the prompt's spec** (deviates from the now-deleted ADR-0003); keeps `pydantic` and `pytest` as the two `[PDF-implied]` exceptions |
| `environment.yml`   | minimal conda mirror with `pip: -r requirements.txt` so it stays in sync with `requirements.txt` forever |
| `.gitignore`        | clean per-§41-directory rules with `.gitkeep` bang-overrides; drops the `[ADDED]` `.pytest_cache`/`.mypy_cache`/`.ruff_cache` lines as those tools are gone (keeps `.pytest_cache` since pytest stays) |
| `README.md`         | strict §41-aligned structure: title + status + Quick start + repo tree + plan + key guardrails + Backup + Citations sections (no ADR mentions, no INSTALL.md cross-reference) |
| `tests/conftest.py` | extend existing fixtures with a `sys.path.insert(PROJECT_ROOT)` block so tests work without `pip install -e .` (pyproject is gone) |
| `progress.md`       | append "Week 2 (continued) — PDF-alignment cleanup" entry at the top |

## Will DELETE (Phase 6)

| Path               | Reason |
|--------------------|--------|
| `CLEANUP_PLAN.md`  | this file; its purpose was the dry-run preview, removed after verification passes |

## Items flagged but not touched

- **LangChain re-introduction in `requirements.txt`.** The prompt's
  Phase 5.1 spec explicitly includes `langchain>=0.2,<0.3` as
  "scaffolding; primary pipeline is plain Python." This directly
  contradicts the now-deleted ADR-0003 ("No LangChain in the RAG
  layer"). I will follow the prompt's stated `requirements.txt`. I
  will note this in `CLEANUP_REPORT.md`.
- **Vision-backbone and embedding deviations.** Once
  `WEEK2_AUDIT.md` is deleted, the only paper trail explaining the
  ResNet50→EfficientNet and MiniLM→BGE swap will be in code (config
  `Literal` fields + Week-2 commits). The deleted `one pager.pdf`
  showed the old choices. If the supervisor expects an updated
  one-pager, that's a follow-up. Flagged in `CLEANUP_REPORT.md`.
- **`src/disease/` ADR-0001 footnote.** None of the `src/` files
  reference the ADRs by filename, so deleting `decisions/` will not
  orphan any imports. I checked.

## Test impact preview

Tests that currently pass and what changes:

- `tests/utils/test_*.py` — no change. seeding, paths, config, logging
  all keep working.
- `tests/{disease,soil,rag,integration,explain,eval}/test_smoke.py` —
  no change. None of them import deleted modules.
- `tests/eval/test_citation_verification.py` — no change.
- `tests/rag/test_prompt_rendering.py` — no change.

I expect `pytest -q` to still report 50 passed after Phase 5.

---

**Stopping here per the prompt's working-style note.** Commit will be `"plan: cleanup preview"`. After human review, resume from Phase 2.
