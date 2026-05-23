# Hugging Face Hub Upload Report — Phase 5 §C

Date: 2026-05-23
Account: `ankit-iiitdmj` (Write token, role=write)
Visibility: **all three repos are PRIVATE**

| Dataset | URL | Train / Val / Test | Classes | Upload time |
|---|---|---:|---:|---:|
| **Paddy Doctor** | https://huggingface.co/datasets/ankit-iiitdmj/iks-paddy-doctor | 8,325 / 1,041 / 1,041 | 10 | 530 s |
| **PlantDoc** | https://huggingface.co/datasets/ankit-iiitdmj/iks-plantdoc | 2,046 / 256 / 256 | **27** (post-§A merge) | 148 s + 1 retry |
| **PlantVillage** | https://huggingface.co/datasets/ankit-iiitdmj/iks-plantvillage | 43,443 / 5,431 / 5,431 | 38 (color variant) | 614 s |

**Total wall-clock:** ~21 min push + ~7 min retry overhead = ~28 min total.
**Total network bandwidth:** ~3 GB push (parquet-packed).

Hub-side verification (programmatic `HfApi().dataset_info`):

```
ankit-iiitdmj/iks-paddy-doctor : siblings=6, private=True
ankit-iiitdmj/iks-plantdoc     : siblings=6, private=True
ankit-iiitdmj/iks-plantvillage : siblings=6, private=True
```

Each repo's 6 siblings = `.gitattributes` + `README.md` (dataset card) +
4 parquet files for train/val/test plus the dataset_info schema.

## Incidents

- **PlantDoc upload v1 stalled at 99% on the second parquet shard**
  (193 MB / 195 MB). Log file size unchanged for 8 minutes; Python
  process alive but TCP wedged. Killed and restarted via the stall
  detector. The retry resumed by skipping already-committed shards
  and finished in 148 s.
- **No other upload incidents.**

## Ready-for-Colab one-liner

Run from a Python REPL on any machine logged into the same HF account
(`huggingface-cli login`):

```python
from datasets import load_dataset
ds = load_dataset("ankit-iiitdmj/iks-plantdoc", split="test")
print(len(ds), ds[0]["image"].size)
# Expected:  256  (some H x W tuple, varies per image)
```

The same recipe works for `iks-paddy-doctor` and `iks-plantvillage`.

## Dataset cards

All three repos now carry a generated `README.md` with:

- YAML frontmatter (`license`, `task_categories: image-classification`,
  `size_categories` bucket).
- One-paragraph dataset description + source citation.
- Per-split sizes table.
- Full class list with indices.
- Preprocessing summary (80/10/10 stratified, seed=42).

For PlantDoc, the dataset card explicitly documents the 28→27 merge
done in §A (Tomato two spotted spider mites leaf folder → Tomato leaf
with `was-spider-mites-` filename prefix).

## Phase 5 next step

The Colab notebook generated in §G will read these via
`datasets.load_dataset("ankit-iiitdmj/iks-<name>")`. Pin the HF Hub
token in Colab via `notebook_login()` (Cell 3 of the notebook).
