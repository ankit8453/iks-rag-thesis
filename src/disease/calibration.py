"""Confidence calibration for the disease classifier (temperature scaling).

Why this exists
---------------
A softmax classifier is usually **over-confident**, and worse so on inputs from
outside its training distribution — exactly the "untrained plant" case we now
want to show a confidence for. Reporting the raw softmax probability to a farmer
would therefore mislead. **Temperature scaling** (Guo et al., 2017) fixes this
with a single scalar ``T``: replace ``softmax(z)`` with ``softmax(z / T)``. It
never changes *which* class wins (so accuracy is untouched), only how confident
the number is. ``T`` is fit once on a held-out set by minimising negative
log-likelihood; ``T > 1`` softens an over-confident model.

Pure NumPy and dependency-free: the fit is a deterministic 1-D search, so it runs
and unit-tests on a laptop. The fitted ``T`` is stored as one number and applied
to the logits the classifier already returns (``DiseasePrediction.logits``) —
the model itself is not modified.
"""

from __future__ import annotations

import numpy as np

#: Identity temperature — no calibration applied. The app falls back to this
#: until a real T is fitted on the test set (on Colab) and stored.
NO_CALIBRATION: float = 1.0


def _as_2d(logits) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    return arr[None, :] if arr.ndim == 1 else arr


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable temperature-scaled softmax. Accepts 1-D or 2-D."""
    z = _as_2d(logits) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[0] if np.asarray(logits).ndim == 1 else probs


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Alias for :func:`softmax` — the calibrated probability distribution."""
    return softmax(logits, temperature)


def top_confidence(logits, temperature: float = 1.0) -> float:
    """Calibrated confidence of the winning class for a single logit vector."""
    return float(np.max(softmax(np.asarray(logits).ravel(), temperature)))


def negative_log_likelihood(logits, labels, temperature: float) -> float:
    """Mean NLL of ``labels`` under the temperature-scaled distribution."""
    probs = softmax(_as_2d(logits), temperature)
    idx = np.asarray(labels, dtype=int)
    picked = probs[np.arange(len(idx)), idx]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(
    logits, labels, *, lo: float = 0.25, hi: float = 10.0, iters: int = 60,
) -> float:
    """Fit the temperature that minimises NLL on ``(logits, labels)``.

    NLL is convex in ``log T`` for temperature scaling, so a golden-section
    search over ``[lo, hi]`` converges reliably and deterministically without
    any optimiser dependency.
    """
    logits = _as_2d(logits)
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return NO_CALIBRATION

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = negative_log_likelihood(logits, labels, c)
    fd = negative_log_likelihood(logits, labels, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = negative_log_likelihood(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = negative_log_likelihood(logits, labels, d)
    return round((a + b) / 2.0, 4)


def expected_calibration_error(
    confidences, correct, *, n_bins: int = 15,
) -> float:
    """ECE — mean gap between confidence and accuracy across probability bins.

    Lower is better; reporting ECE before vs after calibration is the evidence
    that the displayed confidence became trustworthy.
    """
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo_e, hi_e = edges[i], edges[i + 1]
        in_bin = (conf > lo_e) & (conf <= hi_e) if i > 0 else (conf >= lo_e) & (conf <= hi_e)
        if not in_bin.any():
            continue
        ece += abs(hit[in_bin].mean() - conf[in_bin].mean()) * (in_bin.mean())
    return float(ece)


def calibrated_topk(logits, class_names, temperature: float, k: int = 3):
    """Return the top-``k`` ``(class_name, calibrated_prob)`` pairs."""
    probs = softmax(np.asarray(logits).ravel(), temperature)
    order = np.argsort(probs)[::-1][:k]
    return [(class_names[int(i)], float(probs[int(i)])) for i in order]


__all__ = [
    "NO_CALIBRATION",
    "apply_temperature",
    "calibrated_topk",
    "expected_calibration_error",
    "fit_temperature",
    "negative_log_likelihood",
    "softmax",
    "top_confidence",
]
