"""Guardrail behavior for the Phase 10 UI.

Two paths in ``app.guardrail.is_leaf``:

* Native: ``HAS_NO_LEAF_CLASS=True`` — the disease model's own
  ``no_leaf`` class drives the verdict.
* Fallback: ``HAS_NO_LEAF_CLASS=False`` — segmentation foreground
  fraction must sit in the accept band.

Tests stub both surfaces (engine.predict / segment.segment) so no GPU /
HF Hub / rembg dependency leaks into pytest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from PIL import Image


# --------------------------------------------------------------------- #
# Native path — has_no_leaf_class=True
# --------------------------------------------------------------------- #


@dataclass
class _FakePrediction:
    class_name: str
    confidence: float = 0.9
    class_index: int = 0


@dataclass
class _FakeInfResult:
    prediction: _FakePrediction
    top_k: list = None
    gradcam_overlay: Any = None


class _FakeEngine:
    """Just enough surface for ``guardrail.is_leaf`` to call."""

    def __init__(self, class_name: str) -> None:
        self._class_name = class_name

    def predict(self, image: Any) -> _FakeInfResult:
        return _FakeInfResult(prediction=_FakePrediction(class_name=self._class_name))


def test_native_rejects_no_leaf_prediction() -> None:
    from app.guardrail import is_leaf

    img = Image.new("RGB", (64, 64), (10, 200, 30))
    engine = _FakeEngine(class_name="no_leaf")
    ok, reason = is_leaf(img, has_no_leaf_class=True, disease_engine=engine)
    assert ok is False
    assert "no_leaf" in reason.lower()


def test_native_accepts_normal_leaf_prediction() -> None:
    from app.guardrail import is_leaf

    img = Image.new("RGB", (64, 64), (10, 200, 30))
    engine = _FakeEngine(class_name="Tomato leaf")
    ok, reason = is_leaf(img, has_no_leaf_class=True, disease_engine=engine)
    assert ok is True
    assert reason == ""


def test_native_requires_engine() -> None:
    from app.guardrail import is_leaf

    img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError):
        is_leaf(img, has_no_leaf_class=True, disease_engine=None)


# --------------------------------------------------------------------- #
# Fallback path — segmentation-based
# --------------------------------------------------------------------- #


@dataclass
class _FakeSegResult:
    foreground_fraction: float
    mask: Any = None
    flagged_as_failure: bool = False
    method: str = "test"


def _patch_segment(monkeypatch: pytest.MonkeyPatch, fraction: float) -> None:
    """Stub ``src.disease.segment.segment`` to return a fixed fraction."""

    def _fake_segment(image: Any, style: str = "field") -> _FakeSegResult:
        return _FakeSegResult(foreground_fraction=fraction)

    monkeypatch.setattr(
        "src.disease.segment.segment", _fake_segment, raising=True,
    )


def test_fallback_rejects_when_foreground_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sky / soil / empty-frame uploads have <5% leaf foreground."""
    from app.guardrail import is_leaf

    _patch_segment(monkeypatch, fraction=0.02)
    img = Image.new("RGB", (64, 64), (200, 200, 220))
    ok, reason = is_leaf(
        img, has_no_leaf_class=False,
        foreground_min=0.08, foreground_max=0.92,
    )
    assert ok is False
    assert "foreground" in reason.lower()


def test_fallback_rejects_when_foreground_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat single-color upload segments as ~100% foreground — bad."""
    from app.guardrail import is_leaf

    _patch_segment(monkeypatch, fraction=0.98)
    img = Image.new("RGB", (64, 64), (50, 80, 30))
    ok, reason = is_leaf(
        img, has_no_leaf_class=False,
        foreground_min=0.08, foreground_max=0.92,
    )
    assert ok is False
    assert "single solid surface" in reason.lower() or "foreground" in reason.lower()


def test_fallback_accepts_normal_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """40% leaf foreground sits squarely in the accept band."""
    from app.guardrail import is_leaf

    _patch_segment(monkeypatch, fraction=0.40)
    img = Image.new("RGB", (64, 64), (50, 200, 80))
    ok, reason = is_leaf(
        img, has_no_leaf_class=False,
        foreground_min=0.08, foreground_max=0.92,
    )
    assert ok is True
    assert reason == ""


def test_fallback_ignores_engine_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``has_no_leaf_class`` is False the engine is never consulted —
    a passed-in engine that would crash on call must not block the path."""
    from app.guardrail import is_leaf

    _patch_segment(monkeypatch, fraction=0.35)

    class _ExplodingEngine:
        def predict(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover
            raise AssertionError("fallback path must not call engine.predict")

    img = Image.new("RGB", (64, 64), (30, 180, 40))
    ok, _ = is_leaf(
        img, has_no_leaf_class=False, disease_engine=_ExplodingEngine(),
    )
    assert ok is True
