# Soil Cross-Region Split (per master reference §14)

This split operationalises the §14 cross-region validation requirement
for the soil module.

## Layout

- `train.json` — 90% of the Phantom-fs Comprehensive Soil Classification
  dataset, stratified by class, seed = 42.
- `val.json` — 10% of Phantom-fs, stratified by class, seed = 42.
- `test.json` — 100% of the IRSID (Indian-Region Soil Image Dataset)
  Kaggle mirror.
- `soil_cross_region_meta.json` — class maps and provenance metadata.

`SplitEntry.path` is prefixed with the dataset tag (`phantomfs/…` or
`irsid/…`) so loaders can resolve back to the correct raw root.

## Label-space harmonisation note

Phantom-fs labels are **deposit names**:
Alluvial · Arid · Black · Laterite · Mountain · Red · Yellow.

IRSID labels are **texture composition names**:
Clay · Sand · Silt (from the `Type` column in IRSID's
`Practical_Reading.csv`).

These are **categorically different label spaces** — they must not be
merged into a single class index. Two separate class maps live in
`soil_cross_region_meta.json` (`train_class_map` for Phantom-fs,
`test_class_map` for IRSID).

## How Phase 6 will use this split

Phase 6 (soil training + evaluation) will report **two** cross-region
numbers:

1. **Same-distribution accuracy** — train+val on Phantom-fs, evaluate
   on the Phantom-fs val set. Standard intra-dataset metric.
2. **Cross-region transfer stress test** — evaluate the trained soil
   model on IRSID test. Because the label spaces differ, this is not a
   like-for-like accuracy number; it is reported as a *qualitative
   transfer-learning signal*. For example, predictions on IRSID will
   land in the Phantom-fs deposit-label space, and Phase 6 will record
   the distribution of those predictions per IRSID texture class.

The §14 spirit is "model generalises to unseen Indian regions". The
deposit-vs-texture mismatch limits how cleanly we can score that, and
the upgrade path is the full IEEE DataPort IRSID release (see the
`TODO[IEEE]:` block in `scripts/download_irsid.py`).

## Per §14 — sieve-analysis numbers are off-limits

`Practical_Reading.csv` in the IRSID Kaggle mirror also carries
quantitative `Sand`, `Silt`, `Clay` percentages. These are physical
sieve-analysis ground truth and **must NOT be exposed as model targets**
(`[PDF §14]`). Only the categorical `Type` column is used as an
evaluation label here.

## Provenance

- Built by `scripts/build_soil_cross_region.py`, seed = 42.
- Phantom-fs source: Kaggle `ai4a-lab/comprehensive-soil-classification-datasets`
  (`Orignal-Dataset/` subdir only; CyAUG variant deferred to Phase 6).
- IRSID source: Kaggle `kiranpandiri/indian-region-soil-image-dataset`
  (mirror of IEEE DataPort DOI 10.21227/2zz3-f173, 16 samples).
