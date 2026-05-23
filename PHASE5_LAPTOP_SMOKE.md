# Phase 5 Laptop Smoke Test — Results

Date: 2026-05-23
Machine: Ankit's laptop (CPU only — no GPU)
Branch: `cleanup/pdf-alignment`
Wall-clock: **6.1 seconds**

## What was tested

`scripts/phase5_smoke_test.py` exercises every piece the §A–§G work
wired together, end to end, on tiny synthetic inputs. No real training,
no GPU. The intent is to catch wiring bugs before Colab time is spent.

| # | Step | Result |
|---|---|---|
| 1 | Verify each of the 3 disease datasets has a local sample image discoverable via the split JSON | ✓ 3/3 present (plantvillage, plantdoc, paddy_doctor) |
| 2 | Construct fresh `DiseaseClassifier(num_classes=38, pretrained=False, dropout_rate=0.3)` on CPU | ✓ built |
| 3 | Forward pass on a `(1, 3, 96, 96)` random tensor → logits `(1, 38)` | ✓ loss=3.71 |
| 4 | One backward pass + grad sanity check (`.grad is not None` somewhere) | ✓ |
| 5 | Save → reload checkpoint, verify every parameter tensor matches bit-for-bit | ✓ |
| 6 | `DiseaseInferenceEngine` loads the checkpoint, runs `.predict()` on random RGB → `DiseasePrediction` schema OK + top-k of length 5 | ✓ top-1 c23 @ 0.027 (random init expected) |
| 7 | `compute_gradcam(model, image_tensor, target_class=0)` → `(H, W)` float32 in `[0, 1]` | ✓ shape (96, 96), all zeros because the untrained model has no learned gradient direction |

The Grad-CAM heatmap being all zeros is **expected** for an
unpretrained, untrained classifier — there's no meaningful class-
specific gradient signal yet. The Phase 5 Colab training will fix
that; the smoke test only verifies the call doesn't crash and the
output shape is correct.

## What this proves before Colab

- `DiseaseClassifier` + `DiseasePrediction` + `InferenceResult`
  schemas wire up cleanly.
- `freeze_backbone` / `unfreeze_backbone` / `get_feature_extractor`
  hooks work.
- `state_dict()` / `load_state_dict()` round-trip with no key
  mismatches.
- `DiseaseInferenceEngine.predict()` handles `numpy.ndarray` input
  (and per the unit tests, `PIL.Image` + `torch.Tensor` too).
- `compute_gradcam()` resolves the target layer (EfficientNet
  `conv_head`) without timm naming surprises.
- All 3 datasets' split JSONs point at real local files (means HF
  uploads + local data are consistent).

## What does NOT happen here

- No training updates beyond a single backward pass.
- No HF Hub interaction in the smoke test (the inference engine loads
  from a local `.pt`, not from `ankit-iiitdmj/iks-disease-*`).
- No GPU.

For the real training run, follow `PHASE5_COLAB_GUIDE.md`.

## Reproducing

```
python scripts/phase5_smoke_test.py
```

A `_smoke_work/` directory is created for the round-trip checkpoint;
it's `.gitignore`-able (or just delete it after the run — the smoke
test is idempotent and rebuilds it).
