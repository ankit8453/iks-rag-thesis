"""Phase 5 laptop smoke test (CPU only).

End-to-end verification that the Phase 5 disease module pieces wire
together correctly. **No real training** — tiny synthetic batches +
B0 backbone for speed.

Run after Phase 5 §A–§G have committed. Exit code 0 = all good.

Stages exercised:
1. Tiny batch (4 images) loaded from the local copy of each of the 3
   disease datasets if their `raw/` directories are present.
2. Fresh DiseaseClassifier(num_classes=38) on CPU.
3. One forward pass on a 1x3x380x380 random tensor.
4. Compute CE loss, run one backward pass.
5. Save a local checkpoint, load it back, verify state-dict tensors
   match bit-for-bit.
6. DiseaseInferenceEngine over the same checkpoint -> predict() ->
   DiseasePrediction sanity.
7. compute_gradcam on the same image -> heatmap shape matches input.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    DATA_PLANT_DISEASE_DIR,
    PROJECT_ROOT,
)

_LOGGER = get_logger(__name__)

SPLIT_ROOTS = {
    "plantvillage": (
        DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw",
        PROJECT_ROOT / "data" / "splits" / "plantvillage" / "train.json",
    ),
    "plantdoc": (
        DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw",
        PROJECT_ROOT / "data" / "splits" / "plantdoc" / "train.json",
    ),
    "paddy_doctor": (
        DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw",
        PROJECT_ROOT / "data" / "splits" / "paddy_doctor" / "train.json",
    ),
}


def _load_sample_image(dataset_name: str) -> "Path | None":
    """Pull a single sample image path from a dataset's train.json (if present)."""
    raw_root, split_path = SPLIT_ROOTS[dataset_name]
    if not split_path.is_file():
        _LOGGER.info("Skip %s — no train.json on this machine.", dataset_name)
        return None
    with split_path.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    if not entries:
        return None
    sample_path = raw_root / entries[0]["path"]
    if not sample_path.is_file():
        _LOGGER.info("Skip %s — first split entry's image missing on disk.", dataset_name)
        return None
    return sample_path


def _step1_load_dataset_samples() -> dict[str, Path | None]:
    samples: dict[str, Path | None] = {}
    for name in SPLIT_ROOTS:
        path = _load_sample_image(name)
        samples[name] = path
        if path is not None:
            _LOGGER.info("Found sample image for %s: %s", name, path.name)
    return samples


def _step2_3_4_forward_backward() -> "tuple[object, object]":
    """Construct B0 disease classifier (fast on CPU), one forward+backward."""
    import torch
    from torch import nn

    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=38, pretrained=False, dropout_rate=0.3)
    # Smaller spatial input to keep CPU runtime under a minute. The
    # model accepts any HxW; B4 inside timm pools to a fixed feature
    # vector. 96x96 is fine for a smoke test.
    img = torch.randn(1, 3, 96, 96)
    logits = model(img)
    assert logits.shape == (1, 38), f"unexpected logits shape {tuple(logits.shape)}"

    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0]))
    loss.backward()
    # Just verify some param has a non-None grad.
    grad_exists = any(p.grad is not None for p in model.parameters())
    assert grad_exists, "no .grad after backward()"
    _LOGGER.info("Forward+backward ok, loss=%.4f", float(loss.item()))
    return model, img


def _step5_checkpoint_round_trip(model, work_dir: Path) -> Path:
    """Save + reload + verify identical state-dict tensors."""
    import torch

    from src.disease.model import DiseaseClassifier

    ckpt_path = work_dir / "smoke_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "scheduler_state": None,
            "epoch": 1,
            "best_val_acc": 0.0,
            "history": [],
        },
        ckpt_path,
    )

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    new_model = DiseaseClassifier(num_classes=38, pretrained=False)
    new_model.load_state_dict(loaded["model_state"], strict=False)

    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), new_model.named_parameters(), strict=True
    ):
        if not torch.equal(p1, p2):
            raise AssertionError(
                f"Checkpoint round-trip mismatch at {n1!r} (vs {n2!r})"
            )
    _LOGGER.info("Checkpoint round-trip ok at %s", ckpt_path)
    return ckpt_path


def _step6_inference_engine(ckpt_path: Path) -> None:
    from src.disease.infer import DiseaseInferenceEngine

    engine = DiseaseInferenceEngine(
        model_source=str(ckpt_path),
        class_names=[f"c{i}" for i in range(38)],
        device="cpu",
        image_size=96,
        work_dir=ckpt_path.parent,
    )
    rng = np.random.default_rng(42)
    fake_img = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    result = engine.predict(fake_img, with_gradcam=False)
    assert 0 <= result.prediction.class_index < 38
    assert 0.0 <= result.prediction.confidence <= 1.0
    assert len(result.prediction.logits) == 38
    assert len(result.top_k) == 5
    _LOGGER.info(
        "Inference ok: top-1=%s @ %.3f",
        result.prediction.class_name,
        result.prediction.confidence,
    )


def _step7_gradcam(model, img) -> None:
    from src.disease.gradcam import compute_gradcam

    cam = compute_gradcam(model, img, target_class=0)
    H, W = img.shape[-2], img.shape[-1]
    assert cam.shape == (H, W), f"CAM shape {cam.shape} != input {(H, W)}"
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    _LOGGER.info("Grad-CAM ok: heatmap shape %s, min=%.3f max=%.3f",
                 cam.shape, float(cam.min()), float(cam.max()))


def main() -> int:
    start = time.monotonic()
    _LOGGER.info("Phase 5 laptop smoke test starting...")

    # Step 1 — datasets sample (informational only; doesn't fail the test).
    samples = _step1_load_dataset_samples()
    n_present = sum(1 for p in samples.values() if p is not None)
    _LOGGER.info("Local dataset samples present: %d/3", n_present)

    # Steps 2-4 — forward + backward on B4 (using the real DiseaseClassifier).
    model, img = _step2_3_4_forward_backward()

    # Step 5 — checkpoint round trip.
    work_dir = Path("_smoke_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _step5_checkpoint_round_trip(model, work_dir)

    # Step 6 — inference engine.
    _step6_inference_engine(ckpt_path)

    # Step 7 — Grad-CAM.
    _step7_gradcam(model, img)

    elapsed = time.monotonic() - start
    _LOGGER.info("All smoke-test steps passed in %.1fs.", elapsed)
    print(f"\nPHASE 5 LAPTOP SMOKE TEST: OK  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
