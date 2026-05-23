"""Tests for src.disease.infer (Phase 5 §F).

We don't load real HF Hub checkpoints — instead we save a tiny local
checkpoint and load from a Path so the tests are offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_local_checkpoint(tmp_path: Path, num_classes: int = 10) -> Path:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=num_classes, pretrained=False)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": None,
        "scheduler_state": None,
        "epoch": 1,
        "best_val_acc": 0.5,
        "history": [],
    }
    path = tmp_path / "checkpoint_latest.pt"
    torch.save(ckpt, path)
    return path


def test_disease_prediction_schema() -> None:
    from src.disease.model import DiseasePrediction

    p = DiseasePrediction(
        class_index=2, class_name="x", confidence=0.91, logits=[0.1, 0.2, 0.91, 0.05],
    )
    assert p.class_index == 2
    assert p.class_name == "x"
    assert 0 <= p.confidence <= 1
    assert len(p.logits) == 4


def test_inference_engine_loads_local_and_predicts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("PIL")
    from src.disease.infer import DiseaseInferenceEngine

    ckpt_path = _make_local_checkpoint(tmp_path, num_classes=10)
    class_names = [f"disease_{i}" for i in range(10)]

    engine = DiseaseInferenceEngine(
        model_source=str(ckpt_path),
        class_names=class_names,
        device="cpu",
        image_size=64,    # small for speed
        work_dir=tmp_path / "_inf",
    )
    assert engine.num_classes == 10

    # Single image: numpy ndarray.
    import numpy as np
    arr = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    result = engine.predict(arr, with_gradcam=False)
    assert result.prediction.class_index in range(10)
    assert 0.0 <= result.prediction.confidence <= 1.0
    assert len(result.prediction.logits) == 10
    # Top-k: 5 entries with descending probability.
    assert len(result.top_k) == 5
    probs = [p for _, p in result.top_k]
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
    assert result.gradcam_overlay is None


def test_inference_engine_accepts_pil_and_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pil = pytest.importorskip("PIL.Image")
    pytest.importorskip("timm")
    from src.disease.infer import DiseaseInferenceEngine

    ckpt_path = _make_local_checkpoint(tmp_path, num_classes=5)
    engine = DiseaseInferenceEngine(
        model_source=str(ckpt_path),
        class_names=[f"c{i}" for i in range(5)],
        device="cpu",
        image_size=64,
        work_dir=tmp_path / "_inf",
    )
    # PIL
    pil_img = pil.new("RGB", (80, 80), color=(120, 60, 30))
    r1 = engine.predict(pil_img)
    assert r1.prediction.class_index in range(5)
    # torch.Tensor (3, H, W) already-normalised
    tensor = torch.randn(3, 64, 64)
    r2 = engine.predict(tensor)
    assert r2.prediction.class_index in range(5)


def test_class_names_count_must_match_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.infer import DiseaseInferenceEngine

    ckpt_path = _make_local_checkpoint(tmp_path, num_classes=10)
    with pytest.raises(ValueError, match="num_classes=10"):
        DiseaseInferenceEngine(
            model_source=str(ckpt_path),
            class_names=["only", "three", "names"],
            device="cpu",
            image_size=64,
            work_dir=tmp_path / "_inf",
        )
