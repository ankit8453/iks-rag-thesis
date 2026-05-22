# Phase 4 Summary — Dataset Acquisition & Preprocessing

**Branch:** `cleanup/pdf-alignment` (local commits only; not pushed)
**Date:** 2026-05-22
**Total agent time:** ~3 h 30 m (writing + downloads + processing)
**Total disk used:** **5.4 GB** of `data/` (cap was 20 GB)

---

## (a) What got built

### Section A — Download scripts (`scripts/download_*.py`)
Six per-dataset scripts + an orchestrator + an IEEE DataPort stub.
The scripts are idempotent — re-running them when data is already
present is a no-op.

| Script | Source | Target | Status |
|---|---|---|---|
| `download_plantvillage.py` | Kaggle `abdallahalidev/plantvillage-dataset` | `data/plant_disease/plantvillage/raw/` | ok |
| `download_plantdoc.py` | GitHub codeload zip of `pratikkayal/PlantDoc-Dataset` | `data/plant_disease/plantdoc/raw/` | ok |
| `download_paddy_doctor.py` | Kaggle competition `paddy-disease-classification` | `data/plant_disease/paddy_doctor/raw/` | ok |
| `download_phantomfs_soil.py` | Kaggle `ai4a-lab/comprehensive-soil-classification-datasets` | `data/soil/phantomfs/raw/` | ok |
| `download_irsid.py` | Kaggle `kiranpandiri/indian-region-soil-image-dataset` | `data/soil/irsid/raw/` | ok |
| `download_olid_i.py` | Zenodo record `8105154` (smoke sample only) | `data/causation/olid_i/raw/` | smoke sample |
| `download_all.py` | Orchestrator (sequential, never parallel) | — | works |
| `src/utils/ieee_dataport.py` | Stub for IEEE DataPort (institutional access pending) | — | NotImplementedError |

### Section B — Image validation (`src/utils/data_validators.py`)
- `ImageValidationReport` Pydantic model.
- `validate_image_directory` — two-pass PIL check (verify + load).
- `write_corrupt_list` — saves exclusion list under `results/`.
- `scripts/validate_datasets.py` — runs the validator across every
  registered dataset. **0 corrupt files** found across 185,735 images
  scanned.

### Section C — Stratified splits (`src/utils/data_splits.py`)
- `stratified_split` — two-stage 80/10/10 sklearn split, seed = 42.
- `SplitEntry` (Pydantic), `save_split`, `load_split`, `build_class_map`,
  `discover_class_folder_items`.
- `scripts/build_splits.py` — generates splits for every standard
  dataset (excludes IRSID which is the cross-region test set).

### Section D — Soil cross-region split (§14)
- `make_soil_cross_region_split` in `src/utils/data_splits.py`.
- `scripts/build_soil_cross_region.py` — writes
  `data/splits/soil_cross_region/{train,val,test}.json` plus a
  detailed `README.md` explaining the deposit-vs-texture label
  harmonisation issue and the two evaluation numbers Phase 6 will
  report.

### Section E — Channel normalisation (`src/utils/data_stats.py`)
- `ChannelStats` Pydantic model, `compute_channel_stats`,
  `compute_channel_stats_from_paths`, `save_channel_stats`,
  `load_channel_stats`. Vectorised float64 sum/sum_sq accumulator
  (not per-pixel Welford) — runs in minutes, not hours.
- `scripts/compute_channel_stats.py` — six YAML configs under
  `configs/data/<dataset>_norm.yaml`. Subsample cap = 4,000 for the
  large datasets (PlantVillage, Paddy Doctor); statistics converge
  well before that.

### Section F — Augmentation pipelines
Three new modules, each exposing train/eval factories:

- `src/disease/transforms.py` — modest augmentation (HFlip, mild
  colour jitter, ShiftScaleRotate).
- `src/soil/transforms.py` — **heavier** because of the small sample
  size (HFlip + VFlip + RandomRotate90 + wider jitter + GaussianBlur +
  CoarseDropout).
- `src/integration/causation_transforms.py` — **geometric only**.
  Colour augmentations are forbidden (asserted in
  `tests/integration/test_causation_transforms.py`) because OLID I
  labels are hue-cued.

### Section G — Dataset classes
- `JSONIndexedImageDataset` (in `src/disease/dataset.py`, re-used by
  `src/soil/dataset.py`).
- `MultiLabelImageDataset` (in `src/integration/causation_dataset.py`).
- Three disease factories, three soil factories, one OLID factory.
- `_PrefixedCrossRegionDataset` resolves the cross-region path prefix
  (`phantomfs/...` / `irsid/...`).
- Old `PlantVillageDataset` / `PlantDocDataset` / `SoilTypeDataset`
  kept as backwards-compatible aliases.

### Section H — Stats report + EDA notebook
- `scripts/dataset_stats.py` → `results/dataset_stats.md`
  (177 lines: overview, per-class counts, image sizes, norm stats,
  cross-region detail, OLID vocabulary, validation summary).
- `scripts/build_eda_notebook.py` (one-shot generator).
- `notebooks/dataset_eda.ipynb` — verified by
  `jupyter nbconvert --to notebook --execute` (passes).

### Section I — Docs (this file)
- `progress.md` entry under "Phase 4 (Weeks 14–15)".
- `README.md` status line updated.
- This summary.

---

## (b) Per-dataset image counts after validation

| Dataset | Role | Validated | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| PlantVillage | disease pretraining | 54,305 (color variant) | 43,443 | 5,431 | 5,431 |
| PlantDoc | disease real-field eval | 2,558 | 2,046 | 256 | 256 |
| Paddy Doctor | Indian rice disease | 10,407 | 8,325 | 1,041 | 1,041 |
| Phantom-fs Soil | soil primary (Orignal-Dataset/ only) | 1,189 | 951 | 119 | 119 |
| IRSID | soil cross-region test (§14) | 16 | — | — | 16 |
| OLID I (smoke) | causation (C5) | 83 | 65 | 9 | 9 |
| **Soil cross-region** | combined (§14) | — | 1,070 | 119 | 16 |

The PlantVillage walked count over all variants was 162,916 (color +
grayscale + segmented). Splits use only the `color` variant.

All validation reported **0 corrupt files**, so no
`results/corrupt_files_*.txt` artifacts were written.

---

## (c) Cross-region split sanity check

- `train.json` paths all start with `phantomfs/`, never `irsid/`.
- `val.json` paths all start with `phantomfs/`.
- `test.json` paths all start with `irsid/` (16 entries, one per IRSID
  sample file).
- Two separate class maps live in `soil_cross_region_meta.json`:
  - Phantom-fs deposit names (7 classes: Alluvial, Arid, Black,
    Laterite, Mountain, Red, Yellow).
  - IRSID texture names (3 classes from the `Type` column: Clay, Sand,
    Silt).
- Tests in `tests/utils/test_soil_cross_region.py` verify: no path
  leakage between splits; every Phantom-fs class present in both
  train and val.

---

## (d) Anything corrupt or excluded

**None.** 0 corrupt files across all six datasets. The only files
*not* extracted are ~14 entries from the PlantDoc GitHub zip whose
filenames exceed the Windows 260-character path limit — these were
auxiliary `Apple leaf`-style stock photos, not leaf-disease imagery,
and they were skipped at extraction time (not during validation).

---

## (e) Things for Ankit to spot-check before Phase 5

1. **PlantVillage class count.** The README says 38 classes; our
   `class_map.json` confirms 38. ✓
2. **PlantDoc class count.** The prompt and the upstream README cite
   27 classes; our `class_map.json` has 28. The extra class is
   `Tomato two spotted spider mites leaf` — confirm whether this is
   genuine new content or a near-duplicate of an existing class that
   should be merged in Phase 5.
3. **Paddy Doctor competition format.** The Kaggle competition's
   test set has no labels (it's the leaderboard set). Our `test.json`
   is a held-out subset of the **train** images split 80/10/10, NOT
   the competition's test set. This is the right call for our
   internal evaluation; mention it in the thesis if you compare to
   leaderboard scores.
4. **Phantom-fs class deviation.** The prompt's locked decisions list
   the 7 classes as "alluvial/black/clay/red/laterite/peat/yellow"
   but the upstream Kaggle dataset actually has
   "Alluvial/Arid/Black/Laterite/Mountain/Red/Yellow" — `Clay` and
   `Peat` are missing, `Arid` and `Mountain` are extra. The cross-
   region README documents this; supervisor sign-off recommended
   before Phase 6 trains on these labels.
5. **IRSID size.** Only 16 samples in the Kaggle mirror. The §14
   cross-region test is structurally satisfied but statistical power
   is weak. Phase 6 should treat the cross-region accuracy as
   qualitative. Upgrade path: IEEE DataPort full version (see
   `TODO[IEEE]` in `scripts/download_irsid.py`).
6. **OLID I smoke sample only.** Phase 4 fetched
   `class_distribution.xlsx` + `bottle_gourd__part_1.zip` (~186 MB).
   The full Zenodo record is **19 archives totalling ~14 GB**, not
   the single `OLID-I.zip` the prompt assumed. To pull the full set
   in Phase 11, set `OLID_FULL_DOWNLOAD = True` at the top of
   `scripts/download_olid_i.py` and re-run.
7. **`data/splits/plantvillage/`** committed about 54,000 path
   entries across 4 JSON files (~5 MB total). Verify this is OK; if
   you'd rather not track 5 MB of split JSON, we can `.gitignore`
   `data/splits/*/train.json` and reproduce on demand via
   `scripts/build_splits.py`.
8. **`requirements.txt` does NOT include `matplotlib` / `jupyter` /
   `nbformat`.** These are installed in my local env for §H but are
   not declared. Phase 5 will probably need them in the manifest;
   adding now as a `[PDF-implied §41]` block is reasonable.

---

## Deferred to Phase 11

- **OLID I full ~14 GB download** — flip `OLID_FULL_DOWNLOAD = True`
  in `scripts/download_olid_i.py` and re-run.
- **Multi-label expansion in OLID** — `_labels_for()` in
  `causation_dataset.py` currently returns the folder name verbatim
  as a single label. Phase 11 will swap in real multi-label
  expansion once the full archive arrives.
- **OLID label co-occurrence heatmap** — the EDA notebook ships a
  placeholder. Real heatmap waits on the full dataset.
- **Multi-label causation evaluation logic** — Phase 11 will add
  classification metrics that score multi-hot vectors (per-label
  precision/recall + Hamming loss / Jaccard / mAP).

---

## Section commits (one per section, all local)

```
0ab448a  Phase 4 §A: download scripts + raw data acquisition
f36c415  fix(Phase 4 §A): use kaggle Python API instead of `python -m kaggle`
99b0aae  fix(Phase 4 §A): PlantDoc zip download to bypass Windows path issue
5eea876  fix(Phase 4 §A): Paddy Doctor competition slug + OLID I smoke sample
5193a26  Phase 4 §B: image validation utilities + tests + run on all datasets
cef64bc  Phase 4 §C: stratified splits + tests + per-dataset split files
8190de1  Phase 4 §D: soil cross-region split (§14)
aead50a  Phase 4 §E: channel normalisation stats + 6 YAML configs
3e4e330  Phase 4 §F: augmentation pipelines (disease / soil / causation)
4071bee  Phase 4 §G: dataset classes + factories + tests
acca458  Phase 4 §H: dataset_stats.md + dataset_eda.ipynb
(this commit)  Phase 4 §I: docs
```

Plus a chore commit at the top (`2e97622 chore: gitignore kaggle ...`)
and three §A fixup commits documenting the Kaggle API switch, the
PlantDoc zip-download workaround, and the Paddy Doctor competition
slug correction.

**Not pushed.** Inspect `git log --oneline cleanup/pdf-alignment` and
push manually when satisfied.
