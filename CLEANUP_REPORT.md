# Cleanup Report — PDF-Alignment Pass

Branch: `cleanup/pdf-alignment` (pushed; open PR via
https://github.com/ankit8453/iks-rag-thesis/pull/new/cleanup/pdf-alignment)
Base: `main`
Commits (oldest → newest):

```
98e9536  plan: cleanup preview
628a893  fix: paper/thesis nesting per §41
5a3746c  chore: remove [ADDED] engineering hygiene per PDF alignment
a88f9cd  feat: add PDF-required files (§42, §43, §44, environment check)
92d8e41  refactor: rewrite manifests + README + tests config for PDF-only mode
429d41a  verify: cleanup complete, repo matches §41
```

Net: **50/50 tests pass** post-cleanup. `from src.utils ...` imports
clean. Tree matches §41 (extras are `[PDF-implied]` or `[PDF §44]`).

---

## What was removed

### `[ADDED]` engineering hygiene
- `.pre-commit-config.yaml` — pre-commit hook bundle (ruff, black, mypy, file-hygiene)
- `.github/workflows/ci.yml` and `.github/` directory — GitHub Actions matrix CI
- `pyproject.toml` — PEP-621 metadata + ruff/black/mypy/pytest/pydantic-mypy tool blocks
- `requirements-dev.txt` — dev-deps shim (pytest + pydantic survived into `requirements.txt`)
- `.python-version` — pyenv pin
- `decisions/` + ADRs:
  - `0001-efficientnet-backbones.md`
  - `0002-pydantic-over-hydra.md`
  - `0003-no-langchain-in-rag.md`
  - `README.md` (ADR index)

### `[ADDED]` session reports
- `WEEK2_SUMMARY.md`
- `WEEK2_AUDIT.md`
- `WEEK2_PROMPT.md` (was untracked; removed from disk via `rm`)
- `CLEANUP_PROMPT.md` (was untracked; removed from disk via `rm`)
- `CLEANUP_PLAN.md` (Phase 6, after verification — its purpose was the dry-run preview)

### `[ADDED]` setup convenience
- `INSTALL.md` — essential commands folded into the new `README.md` Quick-start

### Stale artifacts
- `one pager.pdf` — superseded by the EfficientNet-B4/B0 + BGE decisions (showed ResNet50 + MiniLM-L6); was untracked, removed via `rm`

### Redundant placeholders
- `paper/.gitkeep`, `thesis/.gitkeep` — replaced by `paper/README.md` and `thesis/README.md`, which keep the dirs tracked

### Total removed
12 tracked files (~971 lines) + 3 untracked artifacts.

---

## What was added

### `[PDF §X]` files
- `BACKUP.md` — 3-2-1 backup strategy (`[PDF §43]`)
- `references.bib` — empty BibTeX master file with Zotero/BetterBibTeX header (`[PDF §42]`)
- `research_journal/weekly/README.md` (`[PDF §44]`)
- `research_journal/weekly/2026-W21.md` — starter weekly entry for this week (`[PDF §44]`)
- `research_journal/monthly/README.md` (`[PDF §44]`)
- `research_journal/monthly/2026-05.md` — starter monthly entry (`[PDF §44]`)

### `[PDF-implied]` files
- `notebooks/00_environment_check.ipynb` — 6-cell smoke test (torch / timm / sentence-transformers / chromadb / `src.utils.seeding` / `src.utils.paths`). Valid JSON, no executed outputs.
- `pytest.ini` — minimal pytest config (`testpaths = tests`, `addopts = -ra --strict-markers`) since pyproject is gone
- `scripts/README.md`, `demo/README.md`, `paper/README.md`, `thesis/README.md`, `notebooks/README.md` — one-line placeholders explaining what each §41-named directory will hold

### Total added
14 files (~153 lines).

---

## What changed

| File | Summary of rewrite |
|---|---|
| `requirements.txt` | Strict §22 alignment, grouped by category (Deep learning / CV / RAG+NLP / Evaluation / Deployment / Numerical / Configuration enforcement / Test runner). Every line carries a `[§22]` or `[PDF-implied]` tag. Drops every `[ADDED]` dev-only line. **Re-introduces `langchain>=0.2,<0.3` as scaffolding** per the cleanup spec — see "Notable deviations" below. |
| `environment.yml` | Replaced with a 7-line conda mirror that pulls everything from `requirements.txt` via `pip: -r requirements.txt`. The two manifests now stay in sync forever. |
| `.gitignore` | Cleaned per-§41-directory rules with `.gitkeep` bang-overrides. Drops `.mypy_cache` and `.ruff_cache` (tools gone); keeps `.pytest_cache` (pytest stays). Tracks `data/splits/`, `corpus/cleaned/`, `corpus/chunks/` once produced. |
| `README.md` | Rewritten around §41: title + supervisor + status + Quick start + the exact §41 directory tree + 14-phase plan pointer + four key guardrails (§14 soil scope, §37 seeding, §11/§14 cross-region validation, §29 weekly summaries) + Backup pointer + Citations pointer. No mention of pyproject, ADRs, INSTALL.md, or dev tooling. |
| `tests/conftest.py` | Prepends `PROJECT_ROOT` to `sys.path` so `from src...` imports resolve without `pip install -e .`. Existing fixtures (`tmp_corpus_dir`, `seeded_rng`, `tiny_dummy_image`, `sample_retrieved_chunks`) unchanged. |
| `tests/utils/test_paths.py` | Swap `test_project_root_contains_pyproject` → `test_project_root_contains_requirements_txt`. Same shape, points at the §41-named root marker that actually exists post-cleanup. |
| `progress.md` | Prepended a "Week 2 (continued) — PDF-alignment cleanup" entry plus the verified post-cleanup repo tree from `find . -maxdepth 2 -type d`. |

### Total changed
7 files.

---

## Notable deviations and risks

1. **LangChain re-introduction (must flag).** The new `requirements.txt`
   contains `langchain>=0.2,<0.3` with the comment "orchestration (note:
   only as scaffolding; primary pipeline is plain Python)". This was
   prescribed verbatim by the cleanup prompt's Phase 5.1 spec, but it
   directly contradicts the deleted ADR-0003 ("No LangChain in the RAG
   layer"). All RAG code under `src/rag/` still uses plain Python
   wrappers (chromadb + rank_bm25 + transformers + sentence_transformers)
   and does not import `langchain`. The dependency is now installed but
   unused. If you intended to keep ADR-0003's stance, drop the langchain
   line from `requirements.txt`. If you intended the cleanup spec to
   override ADR-0003, no further action needed.

2. **Backbone deviation now has no documentation trail.** The
   `one pager.pdf` and ADR-0001 are both gone. The only remaining
   evidence that the project chose EfficientNet-B4/B0 over ResNet50
   (and BGE over MiniLM-L6) is in `src/disease/config.py` and
   `src/rag/config.py` as `Literal[...]` fields. If the supervisor
   later asks "why did you switch backbones?", the git log will be the
   only answer (commits `b28a26f Week 2 §1: Python project metadata`
   and `e8901d2 Week 2 §3: Pydantic config schemas`). Consider
   memorialising the decision in the eventual thesis / paper draft.

3. **Notebook not interactively verified.** `notebooks/00_environment_check.ipynb`
   is valid JSON (verified with `json.load`) but I did not run it
   through Jupyter in this session. The `import chromadb;
   client = chromadb.Client()` cell needs chromadb installed; the
   `import timm; m = timm.create_model(...)` cell needs timm. These
   are in the new `requirements.txt` but the current environment may
   not have them all installed yet. Run `pip install -r requirements.txt`
   then `jupyter notebook notebooks/00_environment_check.ipynb` to
   confirm before merging.

4. **Pydantic mypy plugin gone.** Two `Field(default_factory=...)`
   sites in `src/soil/config.py` and `src/eval/config.py` use
   module-level `_DEFAULT_*` lists specifically to keep mypy happy.
   mypy is no longer a build dependency, so that defensive shape isn't
   needed — but it doesn't hurt and is readable, so I left it alone.

5. **`progress.md` historical content preserved.** Earlier entries
   mention `pyproject.toml`, `INSTALL.md`, `.python-version`, the
   decisions/ ADRs, etc. as things that *were created* during Week 2.
   These are journal entries about what happened then, not active
   instructions, so the grep verification correctly ignored markdown
   matches. If you want a perfectly clean historical record, you'd
   need to amend the Week 2 entry — but that loses the actual trail
   of what was done and then undone, which seems worse.

---

## Not addressed

Items from the audit / prompts that I deliberately left alone:

- **`notes/iks/extensions/` subdirectory for the three optional IKS
  texts** (Kashyapiyakrishisukti, Vishvavallabha, Brihat Samhita).
  Audit flagged this as a gap but the cleanup prompt did not ask for
  it. Add when the literature review reaches them.
- **One-pager refresh / new ADR explaining backbone choice.** Audit
  flagged this too. Not addressed because the cleanup prompt instructed
  me to delete `decisions/` and the one-pager, leaving no canonical
  place for it. Consider adding a "Decisions" section to `progress.md`
  or to the eventual thesis-intro chapter.
- **`results/figures/` `.gitkeep`.** Still present and tracked
  (Week-2 addition). The audit Table C entry called this "harmless but
  removable"; cleanup prompt's "must not delete" list explicitly
  includes everything under `.gitkeep` files in tracked dirs, so I
  kept it.
- **Markdown narrative inside `notes/` and old `research_journal/daily/`
  that references pyproject etc.** Per the cleanup prompt's grep-
  verification rule ("matches inside markdown narrative are fine if
  explaining what was removed"), I did not rewrite historical notes.

---

## How to merge

```
gh pr create --title "PDF-alignment cleanup" --body "$(cat CLEANUP_REPORT.md)"
# or open the PR via the URL above
```

Branch is intentionally left unmerged — please review the diff on
GitHub before merging into `main`.
