# Week 2 PDF-Alignment Audit

> **Authoring note.** I do not have direct access to the full section-
> numbered master reference document. The only PDF currently in the
> repo is `one pager.pdf`, which is a 4-page summary and does not carry
> the §22 / §37 / §41–§44 numbering used in the prompts. This audit is
> built from two anchors that *are* available to me:
>
> 1. The Week 2 prompt's quoted constraints (Locked Stack, the five
>    Hard Guardrails, and section pointers like "Per master reference
>    §14 (Soil Module): disallowed outputs").
> 2. The Part 2 prompt's own one-line descriptions of each section
>    ("§22 Tools, Libraries, Models", "§37 Process mistakes to AVOID:
>    seeding, version control, per-class metric utilities", etc.).
>
> Wherever a classification depends on something only the full PDF
> resolves (e.g. exact wording of §43's "3-2-1 rule"), the row is
> flagged with **[needs PDF verify]**.

---

## Table A — [PDF §X] items (directly required by the master reference)

| File / item | PDF section | Why it's required |
|---|---|---|
| [`README.md`](README.md) | §41 (Code Repository Structure) | Top-level README is the first item in the §41 tree |
| [`progress.md`](progress.md) | §41 + §29 (weekly milestones) | Weekly progress log; named explicitly in the §41 tree |
| [`requirements.txt`](requirements.txt) | §22 (Tools, Libraries, Models) + §41 | Locks the runtime dependency set named in §22 |
| [`environment.yml`](environment.yml) | §41 (conda alternative shown in the tree) | Listed in the §41 tree as the conda counterpart; currently stale, see Inconsistencies below |
| [`.gitignore`](.gitignore) | §41 | Named in the §41 tree |
| [`corpus/raw/`](corpus/raw/), [`cleaned/`](corpus/cleaned/), [`chunks/`](corpus/chunks/), [`vector_db/`](corpus/vector_db/) (`.gitkeep` each) | §41 + §12 | Exactly the four subdirs named in §41; the RAG corpus persistence layer |
| [`data/plant_disease/`](data/plant_disease/), [`soil/`](data/soil/), [`splits/`](data/splits/) | §41 | Exactly the three subdirs named in §41 — fixed in this same commit set, see "fix:" commit |
| [`models/`](models/) (`.gitkeep`) | §41 | Named in the §41 tree (gitignored) |
| [`paper/`](paper/), [`paper/thesis/`](paper/thesis/) | §41 | Named in the §41 tree |
| [`results/`](results/) | §41 | Named in the §41 tree |
| `notebooks/`, `scripts/`, `demo/` (empty directories) | §41 | Named in the §41 tree (currently empty placeholders, see Gaps) |
| [`src/disease/`](src/disease/) (model, dataset, train, infer, gradcam, config) | §41 + §10 (Disease Module) | Module named in §41; backbone choice and 38-class spec from the Week-2 locked stack |
| [`src/soil/`](src/soil/) (model, dataset, train, infer, config) | §41 + §11 / §14 (Soil Module — visual-only) | Module named in §41; the disallowed-outputs validator operationalises §14 |
| [`src/rag/`](src/rag/) (corpus_loader, chunker, embedder, retriever, hybrid, reranker, generator, prompts, config) | §41 + §12 (RAG) | Module named in §41; hybrid dense+sparse retrieval architecture from §12 |
| [`src/integration/`](src/integration/) (context, causation, three strategies, config) | §41 + §13 (Joint Context) + C2 / C5 contributions | Module named in §41; the three ablations are contribution C2 |
| [`src/explain/`](src/explain/) (gradcam, chunk_highlight) | §41 + (Grad-CAM is a named element in the one-pager) | Module named in §41 |
| [`src/eval/`](src/eval/) (cv_metrics, ragas_eval, expert_annotation, citation_verification, config) | §41 + §15 (Evaluation) | Module named in §41; RAGAS+expert-annotation is the §15 evaluation plan |
| [`src/utils/`](src/utils/) (seeding, paths, logging_setup, config) | §41 | Module named in §41 |
| [`src/utils/seeding.py`](src/utils/seeding.py) | §37 ("Not seeding random numbers") | One of the named mistakes §37 says to avoid; this file is the fix |
| [`src/eval/cv_metrics.py`](src/eval/cv_metrics.py) `ClassificationReport` shape | §37 ("per-class metric utilities") | §37 names this as a mistake to avoid; the dataclass enforces per-class + confusion-matrix shape |
| [`literature_tracker.csv`](literature_tracker.csv) | §42 (Citation Management) | One side of §42 — see Gaps for the other side (Zotero/BibTeX) |
| [`research_journal/`](research_journal/) (README, daily/2026-05-20.md) | §44 (Research Journal Practice) | The structure named in §44 |
| `configs/<module>/default.yaml` × 5 | §41 (configs/ named in the tree) + §22 | Holds the §22 tool/version choices in a reviewable, validated form |

---

## Table B — [PDF-implied] items (operationalising a PDF guardrail without being literally named)

This is the most important table for review. None of these are named in
the PDF (as far as I know), but each one is the only sensible way to
satisfy a guardrail the PDF *does* name.

| File / item | Implements which PDF guardrail | Notes |
|---|---|---|
| [`src/utils/paths.py`](src/utils/paths.py) `PROJECT_ROOT`, `CORPUS_*`, `DATA_*`, etc. | §41 (structure) | Single source of truth for paths; without it every script would re-hard-code relative paths and §41's structure would drift |
| [`src/utils/config.py`](src/utils/config.py) `BaseConfig` (`extra="forbid"`, frozen) | §22 + §37 (mistakes — silent config typos) | Pydantic-strict configs catch YAML typos at load time; otherwise §22's tool/version commitments would silently bit-rot |
| [`src/utils/logging_setup.py`](src/utils/logging_setup.py) | §37 (implied — observability for debugging) | Centralises stderr + dated file logging so reruns are inspectable; not literally named but every long-running ML pipeline needs it |
| `src/<module>/config.py` schemas | §22 + per-module sections (§10–§15) | Locks the §22 stack choices into typed schemas; backbone fields are `Literal` so future PRs can't quietly swap them |
| `SoilConfig.disallowed_outputs` + validator | §14 ("Disallowed claims: NPK, pH, fertility, organic matter, chemical composition") | Enforces the soil-scope guardrail at the schema level; a future PR adding `"npk"` as a head fails on import, not at runtime |
| `SoilConfig.cross_region_validation` flag + `SoilTypeDataset.held_out_regions` | §11 (Soil Module evaluation expectation, per Week-2 prompt guardrail #4) | Cross-region split scaffolding for the soil eval; required by the prompt but the schema/dataset hook is an implementation choice |
| [`src/rag/prompts.py`](src/rag/prompts.py) `RAG_PROMPT_TEMPLATE` (cite chunk IDs, refusal mode) | §15 / §12 + §37 (no fabricated citations) | The prompt-level enforcement of the C4 contribution; failing to do this would let the LLM hallucinate citations |
| [`src/eval/citation_verification.py`](src/eval/citation_verification.py) extractor + `CitationReport` | §15 + C4 (Quantitative Grounding Assessment) | Verifies the prompt-level rule actually held; without it C4 has no measurement function |
| `src/rag/hybrid.py` `HybridRetriever` + `src/rag/reranker.py` `CrossEncoderReranker` | §12 (Hybrid Retrieval: Dense + BM25 + Cross-encoder) | The architectural blocks named in the one-pager; class shape lands now even though bodies are Phase 4 |
| `src/integration/causation.py` `CausalPathway` enum + `CausalContext` dataclass | C5 (cause-conditional retrieval; system does not infer cause from images) | The "user-provided" carrier for the causal hypothesis; without this dataclass the integration module would be free to infer cause from imagery |
| `IntegrationConfig.require_causal_context = True` | C5 | Schema-level switch that refuses to run the integration step without a `CausalContext` |
| `tests/utils/test_seeding.py` (reproducibility assertion) | §37 (seeding) | Verifies `set_global_seed(42)` is actually deterministic across torch + numpy |
| `tests/utils/test_config.py` (extra-key rejection) | §37 / §22 | Verifies the strict-extra config behaviour that protects §22 commitments |
| `tests/utils/test_paths.py::test_corpus_dir_contains_no_dataset_paths` | §41 (corpus vs data separation) | Regression guard for the fix in this same commit set |

---

## Table C — [ADDED] items (pure engineering choices, not in PDF)

| File / item | What it is | Could be removed without violating the PDF? |
|---|---|---|
| [`pyproject.toml`](pyproject.toml) | PEP-621 project metadata; single source of truth for deps + tool configs | **Partly.** §22 only names `requirements.txt`; pyproject is modern hygiene. But removing it forces tool configs back into multiple files (`ruff.toml`, `pytest.ini`, etc.). Recommend keep |
| [`requirements-dev.txt`](requirements-dev.txt) | Dev-only deps separated from runtime | Yes — could merge into a single `requirements.txt` if §22 strictly forbids the split |
| [`.python-version`](.python-version) | pyenv / uv version pin | Yes — convenience only |
| [`INSTALL.md`](INSTALL.md) | Setup instructions | Yes — content could fold into `README.md` |
| [`WEEK2_SUMMARY.md`](WEEK2_SUMMARY.md), this file [`WEEK2_AUDIT.md`](WEEK2_AUDIT.md) | Session reports | Yes — once read |
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml) | Pre-commit hooks (ruff, black, mypy, hygiene) | Yes — engineering hygiene only |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions matrix CI | Yes — but recommend keep, see Recommended actions |
| `[tool.ruff]` config in `pyproject.toml` | Linter | Yes |
| `[tool.black]` config in `pyproject.toml` | Formatter | Yes |
| `[tool.mypy]` + Pydantic mypy plugin config | Static type checking | Yes |
| `[tool.pytest.ini_options]` config | Test runner config (testpaths, markers) | The test runner itself is implied by §37; the precise config block is hygiene |
| [`decisions/`](decisions/) (README + ADRs 0001–0003) | Architecture Decision Records | Yes — process choice. None of the three ADRs are named in the PDF |
| [`notes/cv/`](notes/cv/), [`rag/`](notes/rag/), [`xai/`](notes/xai/), [`iks/`](notes/iks/) (28 markdown templates) | Phase-1 foundation-learning notes | **Implied by Phase 1 in §29 / 40-week plan**, but the specific topic split and skeleton are my engineering choice |
| `[project.optional-dependencies] gpu / notebooks / demo` in `pyproject.toml` | Optional extras | Yes — could install everything by default |
| `src/disease/dataset.py` `PLANTVILLAGE_DEFAULT_ROOT` / `PLANTDOC_DEFAULT_ROOT` constants | Default-root convenience for the stub dataset classes | Yes — could omit and force callers to pass `root` explicitly |
| `src/soil/dataset.py` `SOIL_TYPES_DEFAULT_ROOT` | Same idea on the soil side | Yes |
| `src/utils/seeding.py::assert_seed_set()` helper | Quick fail-fast utility | Yes — `set_global_seed` alone satisfies §37 |
| `Generator[int, None, None]` typing import refactor in `tests/conftest.py` | Modern typing import path | Yes |
| `_DEFAULT_SOIL_HEADS` / `_DEFAULT_RAGAS_METRICS` module-level lists | Workaround for mypy's `Literal` widening | Yes — pure typing concession to keep `mypy src/` clean |
| `[tool.pydantic-mypy]` plugin settings | Lets mypy understand Pydantic Field defaults | Yes — only relevant if mypy stays |
| `src/explain/chunk_highlight.py` `HighlightedSpan` / `HighlightedChunk` dataclasses | Span-level answer-to-chunk attribution scaffolding | **Borderline.** Not named in the PDF I have, but maps to "verifiable citation" in the one-pager; reasonable to keep |
| `models/`, `results/figures/`, `results/logs/` `.gitkeep` files (beyond what §41 strictly requires) | Empty-dir placeholders | Yes |

---

## Possible PDF requirements that are NOT yet implemented

Walked through the section list from the Part 2 prompt. The following
are either missing or only partially present:

- [ ] **§42 Citation Management — Zotero / BibTeX placeholder.** A
      `literature_tracker.csv` exists, but there is no
      `references.bib`, no Zotero export `.json`, and no `paper/` -side
      bibliography file. **[needs PDF verify]** whether §42 also
      expects a BibTeX skeleton committed alongside the CSV.
- [ ] **§43 Backup Strategy — 3-2-1 rule.** Not documented anywhere in
      the repo. Recommend a short `BACKUP.md` (or section in
      `progress.md`) covering: (a) local copy, (b) external drive copy,
      (c) GitHub remote copy. **[needs PDF verify]** for the exact
      wording §43 prescribes.
- [ ] **§44 Research Journal Practice — weekly + monthly templates.**
      `research_journal/README.md` describes templates for daily,
      weekly, and monthly cadences but only `daily/2026-05-20.md`
      exists. Add empty `weekly/` and `monthly/` subdirectories with
      one starter template each.
- [ ] **`notebooks/00_environment_check.ipynb`.** The Week 1 README
      promised one; Week 2 did not create it. Phase 1 study expects
      to run code in notebooks; an environment-check notebook is the
      natural smoke test.
- [ ] **`scripts/` is empty.** §41 names it; eventually it will hold
      `train_disease.py`, `train_soil.py`, `run_rag_eval.py`. Empty
      now is fine but worth a placeholder `scripts/README.md` listing
      the scripts that will land in Phases 4–5.
- [ ] **`demo/` is empty.** §41 names it; will hold the Streamlit /
      Gradio app eventually. Same recommendation as `scripts/`.
- [ ] **Three optional IKS extension texts** (Kashyapiyakrishisukti,
      Vishvavallabha, Brihat Samhita) are mentioned in the one-pager
      and Week 2 prompt but not represented in `notes/iks/` or
      `corpus/raw/` placeholders. A `notes/iks/extensions/` subdir
      with one md file per extension would mirror the structure of
      the primary three.

---

## Items that are inconsistent with the PDF

Beyond the `data/` vs `corpus/datasets/` issue already fixed in Part 1,
I see these tensions:

- **Vision backbone — one-pager says ResNet50, Week 2 ships
  EfficientNet-B4/B0.** The `one pager.pdf` "Key Technical Choices"
  table lists *ResNet50 (fine-tuned)*; the Week 2 locked stack and
  every `src/*/config.py` I just wrote pin EfficientNet. ADR-0001
  captures the rationale (EfficientNet-B4 has more headroom on
  PlantVillage; B0 fits the small soil dataset better) but **the
  one-pager itself was not updated**. If the master reference document
  also says ResNet50, this is a real deviation that needs supervisor
  sign-off, not just my ADR.
- **Embedding model — one-pager says MiniLM-L6, Week 2 ships
  BAAI/bge-large-en-v1.5.** The one-pager's "Hybrid Retrieval" box
  reads *Dense (MiniLM-L6) + BM25 + Cross-encoder Re-ranking*. The
  Week 2 `RAGConfig` defaults to BGE-large with BGE-reranker-base.
  Same situation as above: ADR-implicit (the Week 2 locked stack), but
  the public-facing one-pager still names MiniLM-L6.
- **`environment.yml` is stale.** Still references the Week-1 stack
  (LangChain, ResNet50-era versions). Either rewrite to mirror
  `pyproject.toml`'s dependency list, or delete with a note in
  `progress.md`. The `requirements.txt` and `pyproject.toml` agree;
  `environment.yml` is the odd one out.
- **`models/` checkpoint naming.** §41 (per my second-hand reading of
  the prompt) suggests a `models/disease_*.pt` style. Nothing exists
  yet so there's nothing to be wrong, but the eventual
  `DiseaseClassifier.save(...)` should land checkpoints with names
  matching that convention. **[needs PDF verify]** for the exact
  pattern.
- **Config file extension.** Week 2 uses `.yaml`. The PDF, if it
  prescribes anything, likely says the same — flagging only for
  completeness.

---

## Recommended actions

### Keep as-is
- **`ruff` + `black` + `mypy` + `pytest` + `pre-commit` + GitHub
  Actions CI** — strictly engineering hygiene, but they are the only
  reason the audit could be performed at all (every constraint above
  was checkable in part because the typed configs, tests, and lints
  are in place). Removing them would not technically violate the PDF
  but would make every future PR riskier.
- **`pyproject.toml` as the single source of truth** — the PDF's
  `requirements.txt` is regenerated from it, so both are present and
  consistent. Removing pyproject would scatter tool configs across 4–5
  files.
- **Pydantic v2 configs with `extra="forbid"`** — these are the
  enforcement layer for §14, §22, and §37. Recommend keeping even
  though Pydantic itself isn't named in the PDF (see ADR-0002).
- **`decisions/` ADRs** — they exist precisely because the Week-2
  stack deviated from the one-pager. Keep as the supervisor-facing
  paper trail.

### Consider removing
- **`environment.yml`** — stale and inconsistent with `requirements.txt`
  /`pyproject.toml`. Either rewrite or delete; do not leave as-is.
- **`WEEK2_PROMPT.md`** (still untracked, not in repo) — your call
  whether the prompt itself belongs in version control. I'd keep it
  out and link to it from `progress.md` if you want a paper trail.
- **`requirements-dev.txt`** if §22 prefers a single file — currently
  it's a thin shim over `pyproject.toml [dev]`. Removable.
- **`models/`, `results/figures/`, `results/logs/` `.gitkeep`s** if
  §41 expects only the bare `models/` directory. Harmless but
  removable.

### Add to align with PDF
- **`BACKUP.md` (or section in `progress.md`) for §43** — three
  sentences on local + external + remote copies will do.
- **`references.bib` (empty, or with one self-citation) for §42** —
  pairs with the existing `literature_tracker.csv`.
- **`research_journal/weekly/` + `research_journal/monthly/`
  subdirectories with one template each** — `research_journal/README.md`
  describes them but the directories don't exist on disk yet.
- **`notebooks/00_environment_check.ipynb`** — promised by Week 1's
  README, still missing.
- **One-pager refresh** (or a 0004 ADR) reconciling ResNet50 →
  EfficientNet and MiniLM → BGE. The deviations are defensible but
  the public artifact still shows the old choices.
- **Decide on `one pager.pdf` and `WEEK2_PROMPT.md`** (both currently
  untracked) — they should either be committed deliberately or moved
  out of the repo root.
