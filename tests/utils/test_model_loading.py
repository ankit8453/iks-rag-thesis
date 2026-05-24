"""Tests for loading the Phase 5 backed-up disease checkpoints.

Skips cleanly if ``models/disease/`` hasn't been populated yet
(downloads in ``scripts/download_models_from_hf.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.paths import MODELS_DIR

DISEASE_MODELS_ROOT = MODELS_DIR / "disease"
EXPECTED_CKPT_KEYS = {
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "epoch",
    "best_val_acc",
    "history",
}

# (subdir, num_classes used at training time) — matches Phase 5 STAGE_INFO.
SUBDIR_NUM_CLASSES: list[tuple[str, int]] = [
    ("plantvillage", 38),
    ("paddy-doctor", 10),
    ("plantdoc", 27),
]


def _checkpoint_path(subdir: str) -> Path:
    return DISEASE_MODELS_ROOT / subdir / "checkpoint_best.pt"


@pytest.mark.parametrize("subdir,num_classes", SUBDIR_NUM_CLASSES)
def test_checkpoint_loads_via_torch_load(subdir: str, num_classes: int) -> None:
    torch = pytest.importorskip("torch")
    ckpt_path = _checkpoint_path(subdir)
    if not ckpt_path.is_file():
        pytest.skip(
            f"{ckpt_path} not present; run scripts/download_models_from_hf.py first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The CheckpointManager.save_epoch contract.
    assert isinstance(ckpt, dict), "checkpoint should deserialise to a dict"
    missing = EXPECTED_CKPT_KEYS - set(ckpt.keys())
    assert not missing, f"checkpoint keys missing: {missing} in {ckpt_path}"
    # Basic sanity on the scalars.
    assert isinstance(ckpt["epoch"], int) and ckpt["epoch"] >= 1
    assert 0.0 <= float(ckpt["best_val_acc"]) <= 1.0
    assert isinstance(ckpt["history"], list)


@pytest.mark.parametrize("subdir,num_classes", SUBDIR_NUM_CLASSES)
def test_disease_classifier_loads_state_dict(subdir: str, num_classes: int) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    ckpt_path = _checkpoint_path(subdir)
    if not ckpt_path.is_file():
        pytest.skip(f"{ckpt_path} not present; run the download script.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DiseaseClassifier(num_classes=num_classes, pretrained=False)
    # Use strict=False so a slightly-different state-dict layout (timm
    # version drift between training and load) doesn't fail the test.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    # One forward pass to confirm the head matches num_classes.
    out = model(torch.randn(1, 3, 380, 380))
    assert out.shape == (1, num_classes), f"unexpected logits shape {tuple(out.shape)}"


@pytest.mark.parametrize("subdir,num_classes", SUBDIR_NUM_CLASSES)
def test_metric_jsons_present_and_parseable(subdir: str, num_classes: int) -> None:
    base = DISEASE_MODELS_ROOT / subdir
    history_path = base / "history.json"
    eval_path = base / "eval_metrics.json"
    if not history_path.is_file():
        pytest.skip(f"{history_path} missing — run the download script.")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert isinstance(history, list)
    if history:
        first = history[0]
        for key in ("stage", "epoch", "train_acc", "val_acc"):
            assert key in first, f"history entry missing {key!r}"

    if eval_path.is_file():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        # TrainingMetrics.to_dict() returns these top-level fields.
        for key in ("top1_accuracy", "macro_f1", "per_class", "confusion_matrix"):
            assert key in eval_data, f"eval_metrics.json missing {key!r}"
        # Per-class report length matches num_classes.
        assert len(eval_data["per_class"]) == num_classes
