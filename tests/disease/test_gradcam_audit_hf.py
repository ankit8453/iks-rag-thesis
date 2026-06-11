"""Regression test for the Phase 5-R Grad-CAM audit.

The audit was returning ``n_total=0`` on Colab because it walked
``data/plant_disease/plantdoc/raw/`` (a laptop-only path that doesn't
exist in a fresh Colab runtime) and silently skipped every row.
``audit_engine`` now pulls test images from the HF dataset.

This test locks the HF-first behaviour with a stub dataset + stub
engine. No GPU, no network, no real model load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image


class _FakeHFSplit:
    """Minimal HF Dataset stand-in: indexable, has ``len``, yields rows
    with ``"image"`` (PIL) and ``"label_idx"`` / ``"label"``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class _FakeGradCAMResult:
    def __init__(self, pred_idx: int, pred_label: str, conf: float,
                 heatmap: np.ndarray) -> None:
        self.pred_index = pred_idx
        self.pred_label = pred_label
        self.pred_conf = conf
        self.heatmap = heatmap


class _FakeEngine:
    """Just enough surface for the audit code: a ``class_names`` list."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names


def test_audit_engine_pulls_from_hf_and_returns_non_zero_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape as the user's broken Cell 10 output, just stubbed.

    Pre-fix this returned n_total=0 because every row's ``is_file()``
    was False (PlantDoc raw not on disk). Post-fix it walks the HF
    split directly and returns n_total > 0.
    """
    rows = [
        {
            "image": Image.new("RGB", (32, 32), (i * 20, 100, 100)),
            "label": f"label_{i}",
            "label_idx": i % 3,
        }
        for i in range(5)
    ]
    fake_split = _FakeHFSplit(rows)

    def _fake_load(*args, **kwargs):
        return fake_split

    monkeypatch.setattr("datasets.load_dataset", _fake_load, raising=True)

    def _fake_cam(_image, _engine):
        # Heatmap with peak slightly off-center — so we get a mix of
        # central / not-central across rows.
        h = np.zeros((32, 32), dtype=np.float32)
        h[16, 16] = 1.0  # bang on centre → central=True
        return _FakeGradCAMResult(pred_idx=0, pred_label="label_0",
                                   conf=0.8, heatmap=h)

    monkeypatch.setattr(
        "src.disease.gradcam_audit.disease_gradcam",
        _fake_cam,
        raising=True,
    )

    from src.disease.gradcam_audit import audit_engine

    engine = _FakeEngine(class_names=["label_0", "label_1", "label_2"])
    summary = audit_engine("new", engine)

    assert summary.n_total == 5, (
        "audit_engine returned n_total=0 — the HF-first fix regressed; "
        "the loop is still skipping every row."
    )
    # All 5 stubs had the peak at (16, 16) of a 32×32 image — that's
    # the exact centre, central_attention_rate should be 1.0.
    assert summary.central_attention_rate == 1.0
    # row 0 (label_idx=0) matches pred_index=0; rows 1+2 (1, 2)
    # don't match; rows 3+4 (0, 1) — row 3 matches. So 2/5 correct.
    assert summary.n_correct == 2
    assert summary.accuracy_top1 == pytest.approx(0.4, abs=1e-6)
