"""Tests for temperature-scaling calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.disease import calibration as cal


def test_softmax_sums_to_one_and_T_only_softens() -> None:
    logits = np.array([4.0, 1.0, 0.0])
    p1 = cal.softmax(logits, 1.0)
    p2 = cal.softmax(logits, 2.0)
    assert p1.sum() == pytest.approx(1.0)
    # a higher temperature must LOWER the winning confidence, not change the winner
    assert p2.max() < p1.max()
    assert int(p1.argmax()) == int(p2.argmax()) == 0


def test_temperature_never_changes_the_predicted_class() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 13)) * 5
    for T in (0.5, 1.0, 3.0, 8.0):
        base = logits.argmax(axis=1)
        scaled = cal.softmax(logits, T).argmax(axis=1)
        assert np.array_equal(base, scaled), f"T={T} changed the argmax"


def test_fit_temperature_softens_an_overconfident_model() -> None:
    """Sharp, over-confident logits should fit a temperature > 1."""
    rng = np.random.default_rng(1)
    n, c = 400, 13
    labels = rng.integers(0, c, size=n)
    logits = rng.normal(size=(n, c))
    # make it over-confident: inflate the TRUE class a lot, but wrong ~30% of time
    for i, y in enumerate(labels):
        target = y if rng.random() > 0.3 else rng.integers(0, c)
        logits[i, target] += 12.0
    T = cal.fit_temperature(logits, labels)
    assert T > 1.0, f"expected softening temperature, got {T}"


def test_fit_temperature_is_identity_on_already_calibrated_data() -> None:
    """If the data is well-calibrated, T should sit near 1."""
    rng = np.random.default_rng(2)
    n, c = 600, 13
    logits = rng.normal(size=(n, c)) * 1.0
    # labels drawn FROM the model's own distribution => already calibrated
    labels = np.array([rng.choice(c, p=cal.softmax(logits[i], 1.0)) for i in range(n)])
    T = cal.fit_temperature(logits, labels)
    assert 0.7 < T < 1.4, f"expected T near 1, got {T}"


def test_calibration_reduces_expected_calibration_error() -> None:
    rng = np.random.default_rng(3)
    n, c = 500, 13
    labels = rng.integers(0, c, size=n)
    logits = rng.normal(size=(n, c))
    for i, y in enumerate(labels):
        target = y if rng.random() > 0.35 else rng.integers(0, c)
        logits[i, target] += 10.0

    raw = cal.softmax(logits, 1.0)
    correct = raw.argmax(1) == labels
    ece_before = cal.expected_calibration_error(raw.max(1), correct)

    T = cal.fit_temperature(logits, labels)
    calibd = cal.softmax(logits, T)
    ece_after = cal.expected_calibration_error(calibd.max(1), correct)
    assert ece_after < ece_before, f"ECE not improved: {ece_before:.3f} -> {ece_after:.3f}"


def test_top_confidence_and_calibrated_topk() -> None:
    logits = [5.0, 2.0, 1.0, 0.0]
    names = ["a", "b", "c", "d"]
    assert cal.top_confidence(logits, 1.0) == pytest.approx(cal.softmax(logits).max())
    top = cal.calibrated_topk(logits, names, temperature=2.0, k=2)
    assert [n for n, _ in top] == ["a", "b"]
    assert top[0][1] > top[1][1]
    # softened: top-1 calibrated prob is lower than the raw one
    assert top[0][1] < cal.top_confidence(logits, 1.0)


def test_fit_handles_empty_input() -> None:
    assert cal.fit_temperature(np.zeros((0, 3)), []) == cal.NO_CALIBRATION
