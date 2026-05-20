# ADR-0001 — EfficientNet backbones over ResNet50

## Status

Accepted — 2026-05-20. Supersedes the Week 1 ResNet50 plan.

## Context

The Week 1 `README.md` and configs assumed ResNet50 for both the disease
classifier (PlantVillage, ~50k images, 38 classes) and the soil module
(Kaggle Soil Types, ~1,300 images, 5 classes).

Two problems with that plan:

1. **Disease** — PlantVillage is large enough that we are leaving
   accuracy on the table by sticking to ResNet50. EfficientNet-B4 has
   measurably higher top-1 on ImageNet at comparable inference cost
   (per Tan & Le, 2019) and is the dominant backbone in modern plant
   disease literature.
2. **Soil** — at ~1,300 images, ResNet50 (or B4) will overfit.
   EfficientNet-B0 has roughly a tenth of the parameters and is the
   right choice for the small-data regime.

`timm` exposes both backbones with one line each (`timm.create_model(...)`).

## Decision

- **Disease module** uses `efficientnet_b4`, 380px input,
  `num_classes=38`, ImageNet-pretrained.
- **Soil module** uses `efficientnet_b0`, 224px input, multi-task
  heads (one Linear per visual attribute), ImageNet-pretrained.
- Both `Literal["efficientnet_b4"]` / `Literal["efficientnet_b0"]`
  pinned in the Pydantic schemas so future PRs can't quietly swap them.

## Consequences

- The schema-level pin means changing backbones is a config-schema
  change, which forces an ADR.
- Mixing backbones is a small annoyance for shared inference code (the
  preprocessing differs per backbone) but the `AugmentationConfig`
  carries the normalisation stats, so this is contained.
- We commit to the `timm` API rather than `torchvision.models`. If
  `timm` ever lags behind torch we'll need to swap; the wrapper class
  in `src/disease/model.py::DiseaseClassifier` is the only place that
  touches `timm` directly.

## References

- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks. ICML.
- `timm` (Hugging Face PyTorch Image Models).
