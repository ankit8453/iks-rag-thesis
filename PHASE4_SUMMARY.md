# Phase 4 Summary — Dataset Acquisition & Preprocessing (post-fix)

**Branch:** `cleanup/pdf-alignment` (local commits only; not pushed)
**Date:** 2026-05-23 (post Phase-4-fix reconciliation)
**Total disk used:** **30 GB** of `data/` (prompt envelope was 20-25 GB; over by ~5 GB because Sirajganj v2 grew to 4.49 GB vs the prompt's ~500 MB estimate)

---

## (a) What got built

### Section A — Download scripts (`scripts/download_*.py`)
Seven per-dataset scripts (Sirajganj added this round) + an orchestrator
+ an IEEE DataPort stub. All scripts idempotent.

| Script | Source | Target | Status |
|---|---|---|---|
| `download_plantvillage.py` | Kaggle `abdallahalidev/plantvillage-dataset` | `data/plant_disease/plantvillage/raw/` | ok |
| `download_plantdoc.py` | GitHub codeload zip of `pratikkayal/PlantDoc-Dataset` | `data/plant_disease/plantdoc/raw/` | ok |
| `download_paddy_doctor.py` | Kaggle competition `paddy-disease-classification` | `data/plant_disease/paddy_doctor/raw/` | ok |
| `download_phantomfs_soil.py` | Kaggle `ai4a-lab/comprehensive-soil-classification-datasets` | `data/soil/phantomfs/raw/` | ok |
| `download_irsid.py` | Kaggle `kiranpandiri/indian-region-soil-image-dataset` | `data/soil/irsid/raw/` | ok |
| `download_sirajganj_moisture.py` | **[ADDED]** Mendeley DOI 10.17632/skcc44yvvg.2 | `data/soil/sirajganj_moisture/raw/` | ok |
| `download_olid_i.py` | **[switched]** Kaggle `raiaone/olid-i` (was Zenodo) | `data/causation/olid_i/raw/` | **full** |
| `download_all.py` | Orchestrator (sequential, never parallel) | — | works |

Source switches this session:
- **OLID I** moved from Zenodo (19 archives, 14 GB across) to Kaggle
  `raiaone/olid-i` (single zip, ~13.5 GB) — same dataset, same
  authors, much easier packaging.
- **Sirajganj 2025** added net-new. Mendeley signed download URL
  resolved at runtime via the public API.

The Kaggle SDK download helper now has a 3-attempt outer retry loop
(scripts/_download_utils.py) that discards partial zips between
attempts. Needed because Kaggle's internal retries can exhaust on
14 GB transfers when chunked-encoding errors stack.

### Section B — Image validation (`src/utils/data_validators.py`)
Unchanged from the previous Phase-4 commit. The `validate_datasets.py`
runner walks all six datasets via `_dataset_specs.py`.

### Section C — Stratified splits (`src/utils/data_splits.py`)
Unchanged shape. `scripts/build_splits.py` now includes
`sirajganj_moisture` and excludes `olid_i` (multi-label, handled by §B).

### Section D — Soil cross-region split (§14)
Unchanged. Phantom-fs train+val, IRSID test, deposit-vs-texture
harmonisation note in `data/splits/soil_cross_region/README.md`.

### Section E — Channel normalisation (`src/utils/data_stats.py`)
Unchanged shape. Six norm YAMLs in `configs/data/`:
plantvillage / plantdoc / paddy_doctor / phantomfs / irsid /
sirajganj_moisture / olid_i (the latter now from 3,800 train images
at 380×380, replacing the smoke-sample's 65-image stats).

### Section F — Augmentation pipelines
Unchanged. Disease modest, soil heavier, causation geometric-only.
The `causation_transforms.py` colour-augmentation prohibition is
still asserted by the test suite.

### Section G — Dataset classes
- `JSONIndexedImageDataset` (single-label) — unchanged.
- `MultiLabelImageDataset` (OLID) — updated:
  `_labels_for()` now splits compound symptoms (`bottle_gourd__JAS_MIT`
  → `["bottle_gourd", "JAS", "MIT"]`).
- New `make_sirajganj_moisture_loaders` factory in
  `src/soil/dataset.py`. Smoke-tested:
  `make_sirajganj_moisture_loaders(batch_size=8)['train']` yields
  `torch.Size([8, 3, 224, 224])` images and `torch.Size([8])` labels.

### Section H — Stats report + EDA notebook
- `results/dataset_stats.md` (228 lines now, was 177) regenerated with
  Sirajganj as a new row, OLID at full size (4,749 / 23 multi-label),
  and Phantom-fs at its actual 7-class deposits.
- `notebooks/dataset_eda.ipynb` regenerated:
  - Sirajganj section (class distribution, sample images, image-size
    histogram).
  - OLID I section replaced — the smoke-sample placeholder is gone;
    a real **23×23 multi-label co-occurrence heatmap** built from the
    3,800-image train split is in its place, plus a top-5 frequency
    table.
- Verified by `jupyter nbconvert --to notebook --execute …` running
  end-to-end (12.5 MB executed output with embedded figures).

### Section I — Docs (this file)
- `progress.md` entry under "Phase 4 (Weeks 14–15)".
- `README.md` status line continues to point at this summary.
- This rewrite supersedes the original `PHASE4_SUMMARY.md`.

---

## (b) Per-dataset image counts after validation

| Dataset | Role | Validated | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| PlantVillage | disease pretraining | 54,305 (color variant) | 43,443 | 5,431 | 5,431 |
| PlantDoc | disease real-field eval | 2,558 | 2,046 | 256 | 256 |
| Paddy Doctor | Indian rice disease | 10,407 | 8,325 | 1,041 | 1,041 |
| Phantom-fs Soil | soil primary (Orignal-Dataset only) | 1,189 | 951 | 119 | 119 |
| **Sirajganj 2025 [ADDED]** | moisture_appearance head | 1,177 (Before Aug only) | 941 | 118 | 118 |
| IRSID | soil cross-region test (§14) | 16 | — | — | 16 |
| **OLID I [full]** | causation (C5) | **4,749** | **3,800** | **474** | **475** |
| Soil cross-region | combined (§14) | — | 1,070 | 119 | 16 |

Phantom-fs class names (verified against `class_map.json`):
`Alluvial_Soil`, `Arid_Soil`, `Black_Soil`, `Laterite_Soil`,
`Mountain_Soil`, `Red_Soil`, `Yellow_Soil`. No Clay or Peat
(see §e2).

OLID I multi-label vocabulary (23 tags): 8 crops (ash_gourd,
bitter_gourd, bottle_gourd, cucumber, eggplant, ridge_gourd,
snake_gourd, tomato) + 15 symptom tags (DM, EB, FB, IEM, JAS, K, LM,
LS, MIT, Mg, N, PC, PLEI, PM, healthy).

### OLID split JSON convention — `label_idx` is intentionally 0 for every row

OLID is multi-label, so the per-row `label_idx: int` field in
`data/splits/olid_i/{train,val,test}.json` cannot honestly represent
the 2-3 active tags per image. The actual label information is
encoded in the `label` field (e.g. `bottle_gourd__JAS_MIT`); the
multi-label dataset class calls
`src.integration.causation_dataset._labels_for(entry.label)` to expand
the folder name into the per-image multi-hot vector against
`class_map.json`. The `label_idx: 0` value in the JSON is a
placeholder and **must not be read** as a class identity — bit 0 in
the class map is the `DM` tag, but that does not mean every OLID
image is DM. This convention is documented in
`scripts/build_olid_artifacts.py` and in the docstring of
`_save_split_indices`. Sanity check: loading the train split through
`make_olid_loaders` and inspecting the first batch shows the multi-hot
vectors carry the correct tag set per image (e.g.
`ash_gourd__healthy` → bits 14 and 19, vector sum == 2).

Sirajganj v2 had two variants on disk — `Before Augmentation/` (1,177
originals, used) and `After Augmentation/` (11,457 author-pre-augmented
copies, deferred — we augment ourselves at train time).

---

## (c) Cross-region split sanity check

Unchanged from the previous Phase-4 commit (IRSID untouched this
session). All paths in `train.json` and `val.json` start with
`phantomfs/`; all paths in `test.json` start with `irsid/`. No
leakage. Two separate class maps live in
`soil_cross_region_meta.json` (deposit names ≠ texture names — see the
README in the same folder for Phase 6's planned two-number eval).

---

## (d) Anything corrupt or excluded

**OLID full re-validation: not run as a standalone pass this session.**
The Kaggle archive's SHA-256 was verified at download time. The
3,800-image channel-stats pass in `scripts/build_olid_artifacts.py`
loaded every train image with PIL (it warns and continues on any
failure) and emitted no warnings. The remaining 949 val+test images
will be re-validated by `scripts/validate_datasets.py` at the start
of Phase 5.

**Sirajganj:** 1,177 images discovered, all single-label class-folder
layout, no read failures observed during the 941-image norm-stats
pass.

Other five datasets unchanged from the previous Phase-4 commit — 0
corrupt across 185,735 prior-scanned images.

---

## (e1) Soil module scope finalised

**Kept (3 heads):**

| Head | Supervisor | Output |
|---|---|---|
| `soil_type` | Phantom-fs Orignal-Dataset (7 classes) | Alluvial / Arid / Black / Laterite / Mountain / Red / Yellow |
| `moisture_appearance` | Sirajganj 2025 (3 classes) | dry / moderate / wet |
| `texture` | IRSID + §14 mapping | coarse / fine / mixed (USDA labels mapped: see `configs/data/soil_texture_label_mapping.yaml`) |

**Dropped (2 heads):** `surface`, `cover`. Reason: neither carried IKS-
corpus retrieval value per the soil-parameter coverage audit.
Supervisor sign-off received before this session.

Texture mapping (master plan §14):

| IRSID label | §14 head label |
|---|---|
| sand | coarse |
| loamy_sand | coarse |
| clay | fine |
| sandy_loam | mixed |
| loam | mixed |

The mapping is loadable from code via
`src.soil.config.load_texture_label_mapping()`.

---

## (e2) Phantom-fs class verification

The seven actual Phantom-fs Indian deposit labels (verified against
the upstream `Phantom-fs/Soil-Classification-Dataset` repo and the
Sheth et al. 2025 paper, *Engineering Applications of AI*):

| Class | Phantom-fs train | val | test |
|---|---:|---:|---:|
| Alluvial_Soil | 142 | 17 | 18 |
| Arid_Soil | 121 | 15 | 15 |
| Black_Soil | 130 | 17 | 16 |
| Laterite_Soil | 132 | 17 | 16 |
| Mountain_Soil | 138 | 17 | 18 |
| Red_Soil | 144 | 18 | 18 |
| Yellow_Soil | 144 | 18 | 18 |

The Phase 4 prompt's mention of "Clay" and "Peat" was incorrect —
those folders do not exist in the upstream dataset and never did. The
four "extras" (Arid, Laterite, Mountain, Yellow) are valid Indian
deposit types covering more of the country: laterite = Kerala/Karnataka,
mountain = Himalayan, arid = Rajasthan, yellow = central India.
Supervisor sign-off received before this session.

---

## (e3) PlantDoc class verification

The upstream PlantDoc GitHub repo currently has **28** top-level class
folders. The Singh et al. 2020 paper (CODS-COMAD) cites **27**.

The 28th class is `Tomato two spotted spider mites leaf`. Image counts
this round:

| Tomato class | Images |
|---|---:|
| Tomato Early blight leaf | 87 |
| Tomato Septoria leaf spot | 151 |
| Tomato leaf | 63 |
| Tomato leaf bacterial spot | 110 |
| Tomato leaf late blight | 111 |
| Tomato leaf mosaic virus | 54 |
| Tomato leaf yellow virus | 75 |
| Tomato mold leaf | 91 |
| **Tomato two spotted spider mites leaf** | **2** |

**Verdict:** vestigial folder. With 2 images vs 54-151 in the other
tomato classes, the 28th class is almost certainly two stray images
the GitHub repo never cleaned up after the published 27-class set was
finalised, not a deliberate post-paper addition.

Phase 5 decision pending: drop the class, merge it into a neighbour
(e.g. "Tomato leaf"), or keep with an explicit 0-shot note. Data was
NOT modified this session per the prompt's instruction.

---

## (e) Things for Ankit to spot-check before Phase 5

1. **Sirajganj v2 size discrepancy** — the prompt expected ~500 MB /
   1,177 images. Actual archive is 4.49 GB; the extra ~4 GB is the
   `After Augmentation/` variant (11,457 author-pre-augmented copies)
   which we explicitly do NOT use. Free disk if you want by deleting
   `data/soil/sirajganj_moisture/raw/Soil_Moisture_Dataset/After
   Augmentation/` — it's safe; the split JSONs reference only
   `Before Augmentation/` paths.
2. **OLID multi-label expansion sanity** — open
   `data/splits/olid_i/class_map.json` and confirm the 23 entries are
   the union you expected (8 crops + 15 symptoms). Sample any
   `train.json` row and confirm `_labels_for(row.label)` gives a
   plausible 2-3 tag expansion.
3. **PlantDoc class call** — see (e3). Decide in Phase 5 whether to
   drop, merge, or keep the 2-image spider-mites class.
4. **Phantom-fs 7-class confirmation** — sanity-check that the
   `valid_soil_types` list in `src/soil/config.py` matches what the
   supervisor signed off on.
5. **Soil-texture mapping** — review
   `configs/data/soil_texture_label_mapping.yaml`. The "mixed"
   assignment for both `sandy_loam` and `loam` is the §14 default but
   could be tightened in Phase 6.
6. **IRSID still 16 images** — Phase 5 should treat the cross-region
   accuracy as qualitative. Upgrade path remains the IEEE DataPort
   full version (see `TODO[IEEE]` in `scripts/download_irsid.py`).
7. **`requirements.txt` deps** — confirm the new `matplotlib`,
   `jupyter`, `nbformat`, `iterative-stratification`, `requests`,
   `kaggle` entries are acceptable. They're all `[PDF-implied §41]`.
8. **EDA notebook render** — open `notebooks/dataset_eda.ipynb` and
   skim. The OLID multi-label heatmap should be a 23×23 image with
   clear block-diagonal structure (each image's 2-3 tags co-occur).
9. **PHASE4_FIX_PROMPT.md and .claude/ are untracked** in the working
   tree but not yet committed — your call whether to track or ignore.

---

## Deferred to later phases

- **(i) IRSID full version** pending IIITDM IEEE access. `TODO[IEEE]`
  block in `scripts/download_irsid.py`. If access is denied, the
  `texture` head may have to be dropped in a Phase 6 follow-up.
- **(ii) Phantom-fs CyAUG variant** deferred to Phase 6 if the
  Original-Dataset's 1,189 images prove insufficient for the 7-class
  head's val accuracy.
- **(iii) Sirajganj `After Augmentation/` (~11,500 pre-augmented
  copies)** — same logic as CyAUG; deferred. Our own
  `src/soil/transforms.py` handles augmentation at training time.
- **(iv) Causation-conditional retrieval evaluation logic** —
  Phase 11.
- **(v) Multi-label classification metric utilities** (per-tag
  precision/recall, Hamming loss, mAP) — Phase 11.

---

## Section commits this session

```
28918ff  Phase 4 fix §A: delete OLID I smoke-sample data + derived artifacts
45ace74  Phase 4 fix §B: OLID I full download from Kaggle + real multi-label expansion
0f9548a  Phase 4 fix §C: Sirajganj 2025 moisture dataset for moisture_appearance head
06f6f71  Phase 4 fix §D: soil module scope = soil_type + moisture + texture; Phantom-fs 7-class actual
cf0a8b1  Phase 4 fix §E: requirements.txt notebook deps + PlantDoc class verification
(this commit)  Phase 4 fix §F: regenerate stats + EDA + summary for finalised scope
```

(Plus the original Phase 4 section commits §A–§I below them, untouched.)

---

**Not pushed.** Inspect `git log --oneline cleanup/pdf-alignment` and
push manually when satisfied.
