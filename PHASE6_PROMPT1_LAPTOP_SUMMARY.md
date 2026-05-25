# Phase 6 Prompt 1 — Laptop Soil Data Prep + HF Hub Upload (Summary)

Date: 2026-05-25. Branch: `cleanup/pdf-alignment`. Author: Ankit Pawar.

## Mission

Per `PHASE6_PROMPT1_LAPTOP_PREP.md`: build stratified 80/10/10 parquet
splits of four laptop-side soil datasets, merge IRSID + VIT into a single
texture-axis dataset, push three private HF Hub dataset repos, and
compute combined channel-norm stats for Phase 6 multi-task training.

No model training in this session — that's Prompt 2 of 2 (Colab).

---

## HF Hub state after this session

All three repos are **private** under `ankit-iiitdmj/` and contain
`train` / `val` / `test` parquet splits plus a `README.md` dataset card.

| Repo | Head | Classes | Images | Splits (train/val/test) | Parquets | Total size |
|---|---|---:|---:|---|---:|---:|
| `iks-soil-phantomfs` | `soil_type` | 7 | 1,188 | 951 / 119 / 119 | 6 | 397 MB |
| `iks-soil-sirajganj-moisture` | `moisture_appearance` | 3 | 1,177 | 941 / 118 / 118 | 3 | 188 MB |
| `iks-soil-texture-irsid-vit` | `texture` | 3 | 279 | 223 / 28 / 28 | 3 | 27 MB |

Each parquet row: `image` (HF `Image()` column), `label_idx` (int),
`class_name` (str), `source` (str — `phantomfs` / `sirajganj` / `irsid`
/ `vit` for per-source ablation).

The class index is locked at:

```yaml
# soil_type
0: Alluvial   1: Arid       2: Black     3: Laterite
4: Mountain   5: Red        6: Yellow

# moisture_appearance (Sirajganj's "Moderate" renamed to "moist" per §14)
0: wet        1: moist      2: dry

# texture (USDA-collapsed from IRSID + VIT raw labels)
0: coarse     1: fine       2: mixed
```

## Channel norm stats

Computed over the union of train splits across all three HF Hub repos
(2,114 images at 224×224). Written to `configs/data/soil_norm.yaml`:

```yaml
mean: [0.534532, 0.459212, 0.399999]
std:  [0.216268, 0.199672, 0.209751]
n_images: 2114
image_size: 224
```

## Multi-task supervision

Each row supervises **one** head; the other two heads' labels are filled
with `-1` (ignored by `torch.nn.CrossEntropyLoss`) at DataLoader time
via `src.soil.dataset.build_multitask_labels()`. The Phase 6 training
notebook (Prompt 2) wires this up.

---

## Files added / changed this session

- **Scripts**
  - `scripts/prepare_soil_hf_datasets.py` — three preparers + push helper
  - `scripts/upload_texture_only.py` — thin wrapper for one-repo-at-a-time pushes
  - `scripts/compute_soil_norm_stats.py` — channel stats over HF Hub train splits
- **Configs**
  - `configs/data/soil_soil_type_classes.yaml`
  - `configs/data/soil_moisture_classes.yaml`
  - `configs/data/soil_texture_classes.yaml`
  - `configs/data/soil_norm.yaml`
- **Source**
  - `src/soil/dataset.py` — `build_multitask_labels()` helper
- **Tests**
  - `tests/data/test_soil_hf_datasets.py` — 12 tests, all passing
- **Docs**
  - `progress.md` — Phase 6 prep entry appended
  - `PHASE6_PROMPT1_LAPTOP_SUMMARY.md` — this file

`.gitignore` updated: `_phase6_prep/` (local parquet scratch, no longer
used after the rewrite to `Dataset.push_to_hub`).

---

## What went wrong, and the fix

The first three upload attempts of this session burned ~4 hours before
the final approach worked. Worth documenting so it doesn't happen again.

1. **Attempt 1 — PNG re-encoding into pandas → pyarrow OOM.** The
   prompt's Section B literally said "encodes to PNG bytes for parquet
   storage". I followed that, accumulated PNG bytes in a giant pandas
   DataFrame, and called `df.to_parquet()`. PNG of a camera-shot JPG
   bloats 3–5×, so phantomfs's train parquet hit 824 MB. pyarrow then
   OOM'd on the sirajganj write because it needed a 2.3 GB intermediate
   buffer for the single-shard concat. Phantomfs uploaded with the
   bloat; sirajganj crashed; texture never started.

2. **Attempt 2 — JPEG q=90 in pandas, still `api.upload_file`.** Swapped
   PNG→JPEG to halve the bloat, but kept the single-shard manual
   parquet write + `huggingface_hub.upload_file()` upload path.
   Sirajganj train shard was now 1.67 GB. `api.upload_file` started
   pushing at 2.4 MB/s, then hung for 37 min on a single LFS commit
   with no resume. HF Hub showed only the initial `.gitattributes`
   commit on sirajganj.

3. **Attempt 3 — `Dataset.push_to_hub` with file paths, no resize.**
   Switched to the proven Phase 5 pattern (`datasets.Dataset.from_list`
   + `cast_column("image", Image())` + `DatasetDict.push_to_hub`).
   Phantomfs uploaded cleanly in 6 shards (auto-sharded). Sirajganj
   started uploading at ~2.6 MB/s but the python process **silently
   died** at 88 MB / 446 MB into the upload — and at 281 MB on an
   earlier rerun. Reproducible silent crash on sustained transfers of
   large single shards.

4. **Attempt 4 — `Dataset.push_to_hub` with 768-max-dim resize.** Added
   `_resized_jpeg_bytes()` (PIL resize-so-max(W,H)==768, JPEG q=90)
   before encoding rows. Sirajganj's parquet dropped from 450 MB to
   ~70 MB, uploaded in <1 min. Texture (already small) finished in
   seconds. **This is the version of the script that's committed.**

The lesson — pushed to memory in `feedback_hf_dataset_uploads.md` so
future sessions don't repeat it — is:

> For HF Hub image-classification dataset uploads in this repo,
> always use `Dataset.from_list → cast_column("image", Image()) →
> push_to_hub`. Never hand-roll pre-encoded bytes + pandas parquet +
> `api.upload_file`. If the source images are camera-resolution,
> resize to a max-dim that keeps single parquet shards under ~200 MB.

---

## End checks (per the prompt's Section "End Checks")

- [x] `HfApi().list_repo_files("ankit-iiitdmj/iks-soil-phantomfs", repo_type="dataset")` returns parquets + README.
- [x] Same for `iks-soil-sirajganj-moisture` and `iks-soil-texture-irsid-vit`.
- [x] `configs/data/soil_norm.yaml` exists with valid 3-vector mean/std.
- [x] All three `soil_*_classes.yaml` configs exist.
- [x] `pytest tests/data/test_soil_hf_datasets.py -q` → 12 passed.
- [x] No `git push` executed.
- [x] All three HF Hub dataset repos are PRIVATE.

---

## What's next

Phase 6 Prompt 2 (Colab training notebook):

- EfficientNet backbone (B0 per the existing `src/soil/config.py`'s
  strict `Literal["efficientnet_b0"]`; the prompt's "use B4 unless
  §22 strict" is overridden by the type-checked config).
- 3 heads: `soil_type` (7), `moisture_appearance` (3), `texture` (3).
- Per-sample loss masking via `build_multitask_labels()`.
- Train on union of the three HF Hub datasets; resize-augment-normalise
  using `soil_norm.yaml`.
