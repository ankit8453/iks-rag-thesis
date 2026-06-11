"""Phase 5-R Grad-CAM central-attention audit.

Re-applies the Phase 9 ``central 60 %`` test that originally exposed
the shortcut bias: for every PlantDoc test image, run Grad-CAM (using
:func:`src.explain.gradcam.disease_gradcam`), find the argmax pixel of
the heatmap, check whether it falls inside the central
``CENTRAL_FRACTION`` × ``CENTRAL_FRACTION`` box of the image. The
fraction of images where it does is the **central-attention rate** —
the headline metric for the Phase 5-R keep-or-revert decision.

A *higher* central-attention rate means the model is attending more to
the leaf and less to corners / watermarks / backgrounds — the exact
shortcut bias we are trying to break.

Two engines:

- old: ``ankit-iiitdmj/iks-disease-plantdoc`` (Phase 5 baseline).
- new: the Phase 5-R checkpoint under
  ``models/disease_r/iks-disease-r-plantdoc/checkpoint_best.pt``.

Run both, report the delta. The audit is read-only and never touches
either checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.disease.infer import DiseaseInferenceEngine
from src.explain.gradcam import disease_gradcam
from src.utils.data_splits import load_split
from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    import numpy as np  # noqa: F401

_LOGGER = get_logger(__name__)

# Central box (matches the Phase 9 + Part 1 audit thresholds).
CENTRAL_FRACTION: float = 0.60
CONF_THRESHOLD: float = 0.85

PLANTDOC_RAW_ROOT: Path = (
    PROJECT_ROOT / "data" / "plant_disease" / "plantdoc" / "raw"
)
PLANTDOC_TEST_SPLIT: Path = (
    PROJECT_ROOT / "data" / "splits" / "plantdoc" / "test.json"
)
DEFAULT_OLD_REPO: str = "ankit-iiitdmj/iks-disease-plantdoc"
DEFAULT_NEW_CKPT: Path = (
    PROJECT_ROOT / "models" / "disease_r" / "iks-disease-r-plantdoc"
    / "checkpoint_best.pt"
)


@dataclass
class CAMRowResult:
    rel_path: str
    label: str
    pred_label: str
    pred_conf: float
    peak_y: float
    peak_x: float
    central: bool
    correct: bool


@dataclass
class AuditSummary:
    """Per-engine summary across the PlantDoc test set."""

    engine_name: str
    n_total: int
    n_correct: int
    n_high_conf: int
    n_central: int
    n_correct_and_central: int
    central_attention_rate: float       # n_central / n_total
    correct_central_rate: float         # n_correct_and_central / n_total
    accuracy_top1: float                # n_correct / n_total
    rows: list[CAMRowResult]

    def to_json(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "n_high_conf": self.n_high_conf,
            "n_central": self.n_central,
            "n_correct_and_central": self.n_correct_and_central,
            "central_attention_rate": self.central_attention_rate,
            "correct_central_rate": self.correct_central_rate,
            "accuracy_top1": self.accuracy_top1,
            "rows": [
                {
                    "rel_path": r.rel_path, "label": r.label,
                    "pred_label": r.pred_label, "pred_conf": r.pred_conf,
                    "peak_y": r.peak_y, "peak_x": r.peak_x,
                    "central": r.central, "correct": r.correct,
                }
                for r in self.rows
            ],
        }


# --------------------------------------------------------------------- #
# Per-engine sweep
# --------------------------------------------------------------------- #


def _peak_position(heatmap: Any) -> tuple[float, float]:
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(heatmap, dtype=float)
    flat = int(np.argmax(arr))
    h, w = arr.shape[:2]
    y, x = divmod(flat, w)
    return (float(y) / max(1, h - 1), float(x) / max(1, w - 1))


def _is_central(y: float, x: float, fraction: float = CENTRAL_FRACTION) -> bool:
    half = fraction / 2
    return (0.5 - half) <= y <= (0.5 + half) and (0.5 - half) <= x <= (0.5 + half)


def audit_engine(
    engine_name: str,
    engine: DiseaseInferenceEngine,
    *,
    log_every: int = 25,
    dataset_repo: str = "ankit-iiitdmj/iks-plantdoc",
    split: str = "test",
) -> AuditSummary:
    """Run Grad-CAM over the PlantDoc test set with ``engine`` and tally.

    Pulls the test images from the HF dataset (``dataset_repo``) so
    the audit works on a fresh Colab runtime where the local raw/
    tree doesn't exist. Previously walked ``data/plant_disease/plantdoc/raw/``
    via the Phase-4 split JSON, which always returned zero hits on Colab
    (each path's ``is_file()`` was False and the loop's `continue` ran
    every iteration — silently producing ``n_total=0``).

    Labels are compared by integer ``label_idx`` rather than string
    ``label`` so the comparison survives the NEW model's 28-class
    head (27 PlantDoc + 1 no_leaf reject); the first 27 indices align
    with the OLD model's 27-class head.
    """
    from datasets import load_dataset  # noqa: PLC0415

    _LOGGER.info(
        "[%s] pulling %s split=%s from HF for audit ...",
        engine_name, dataset_repo, split,
    )
    hf_split = load_dataset(dataset_repo, split=split)
    total = len(hf_split)

    # Per-engine: best-effort name→idx map so we can match the engine's
    # string ``pred_label`` against the ground-truth integer label_idx.
    name_to_idx: dict[str, int] = {}
    for name in getattr(engine, "class_names", []) or []:
        if name not in name_to_idx:
            name_to_idx[name] = len(name_to_idx)

    rows: list[CAMRowResult] = []
    for i in range(total):
        row = hf_split[i]
        try:
            pil = row["image"].convert("RGB")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("HF row %d: image decode failed: %s", i, exc)
            continue
        gt_idx = int(row["label_idx"])
        gt_label = str(row.get("label", f"class_{gt_idx}"))
        try:
            cam = disease_gradcam(pil, engine)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("[%s] CAM failed on row %d: %s", engine_name, i, exc)
            continue
        y, x = _peak_position(cam.heatmap)
        central = _is_central(y, x)
        # Compare by integer index: engine's argmax pred_index vs HF
        # row's label_idx. This is robust to (a) the new 28-class head
        # appending no_leaf at idx 27 and (b) class_<i> fallback labels.
        correct = (int(cam.pred_index) == gt_idx)
        rows.append(CAMRowResult(
            rel_path=str(i),
            label=gt_label,
            pred_label=cam.pred_label,
            pred_conf=float(cam.pred_conf),
            peak_y=y, peak_x=x,
            central=central, correct=correct,
        ))
        if (i + 1) % log_every == 0:
            n_central = sum(1 for r in rows if r.central)
            _LOGGER.info(
                "[%s] %d/%d done — central so far: %d (%.0f%%)",
                engine_name, i + 1, total, n_central,
                100 * n_central / max(1, len(rows)),
            )

    n_total = len(rows)
    n_correct = sum(1 for r in rows if r.correct)
    n_high = sum(1 for r in rows if r.pred_conf >= CONF_THRESHOLD)
    n_central = sum(1 for r in rows if r.central)
    n_cc = sum(1 for r in rows if r.correct and r.central)
    denom = max(1, n_total)
    return AuditSummary(
        engine_name=engine_name,
        n_total=n_total, n_correct=n_correct, n_high_conf=n_high,
        n_central=n_central, n_correct_and_central=n_cc,
        central_attention_rate=n_central / denom,
        correct_central_rate=n_cc / denom,
        accuracy_top1=n_correct / denom,
        rows=rows,
    )


# --------------------------------------------------------------------- #
# OLD vs NEW driver
# --------------------------------------------------------------------- #


def run_old_vs_new(
    *,
    old_engine: DiseaseInferenceEngine | None = None,
    new_engine: DiseaseInferenceEngine | None = None,
    device: str = "cpu",
) -> tuple[AuditSummary, AuditSummary]:
    """Build the two engines (if not provided) and run the audit on both.

    Returns ``(old_summary, new_summary)``.

    The OLD engine pulls from HF (``DEFAULT_OLD_REPO``); the NEW engine
    loads the local checkpoint at ``DEFAULT_NEW_CKPT``. If the NEW
    checkpoint is absent (Phase 5-R Part 2 has not been run on this
    machine yet), the new audit is skipped and ``new_summary`` returns
    an empty :class:`AuditSummary` so the caller can render the OLD
    baseline alone.
    """
    if old_engine is None:
        _LOGGER.info("Loading OLD engine from %s ...", DEFAULT_OLD_REPO)
        old_engine = DiseaseInferenceEngine(
            model_source=DEFAULT_OLD_REPO, device=device,
        )
    if new_engine is None:
        if not DEFAULT_NEW_CKPT.is_file():
            _LOGGER.warning(
                "NEW checkpoint not found at %s — skipping NEW audit. "
                "Run the cascade first (notebooks/phase5r_retrain.ipynb).",
                DEFAULT_NEW_CKPT,
            )
            new_summary = AuditSummary(
                engine_name="new (missing)", n_total=0, n_correct=0,
                n_high_conf=0, n_central=0, n_correct_and_central=0,
                central_attention_rate=0.0, correct_central_rate=0.0,
                accuracy_top1=0.0, rows=[],
            )
            old_summary = audit_engine("old", old_engine)
            return old_summary, new_summary
        _LOGGER.info("Loading NEW engine from %s ...",
                     DEFAULT_NEW_CKPT.relative_to(PROJECT_ROOT))
        new_engine = DiseaseInferenceEngine(
            model_source=str(DEFAULT_NEW_CKPT), device=device,
        )

    old_summary = audit_engine("old", old_engine)
    new_summary = audit_engine("new", new_engine)
    return old_summary, new_summary


def print_comparison(old: AuditSummary, new: AuditSummary) -> None:
    """Print the OLD vs NEW comparison table the notebook's verdict
    cell reads."""
    print()
    print("=" * 78)
    print("Phase 5-R Grad-CAM audit — central-attention rate (central 60% box)")
    print("=" * 78)
    print(f"{'metric':<32} {'OLD':>14} {'NEW':>14}  {'delta':>10}")
    print("-" * 78)

    def row(name: str, old_v: float, new_v: float, pct: bool = True) -> None:
        if pct:
            print(
                f"{name:<32} {old_v:>13.1%} {new_v:>13.1%}  "
                f"{(new_v - old_v):>+9.1%}"
            )
        else:
            print(
                f"{name:<32} {old_v:>14.4f} {new_v:>14.4f}  "
                f"{(new_v - old_v):>+10.4f}"
            )

    row("central-attention rate", old.central_attention_rate, new.central_attention_rate)
    row("correct AND central", old.correct_central_rate, new.correct_central_rate)
    row("top-1 accuracy", old.accuracy_top1, new.accuracy_top1)
    print()
    print(f"n_total: OLD={old.n_total}  NEW={new.n_total}")
    print("=" * 78)


def keep_or_revert(old: AuditSummary, new: AuditSummary,
                   *, min_attention_gain: float = 0.05,
                   max_accuracy_drop: float = 0.03) -> str:
    """Apply the prompt's keep/revert rule.

    Default thresholds:

    - central attention rate must improve by at least 5 absolute
      percentage points;
    - top-1 accuracy must not drop by more than 3 absolute pp.

    Returns ``"keep"`` or ``"revert"`` plus an explanation string.
    """
    if new.n_total == 0:
        return "revert (NEW checkpoint missing — re-run training before deciding)"
    delta_att = new.central_attention_rate - old.central_attention_rate
    delta_acc = new.accuracy_top1 - old.accuracy_top1
    if delta_att >= min_attention_gain and delta_acc >= -max_accuracy_drop:
        return (
            f"keep (central attention +{delta_att:.1%}, accuracy "
            f"{delta_acc:+.1%}; gain >= +{min_attention_gain:.0%}, "
            f"drop <= {max_accuracy_drop:.0%})"
        )
    return (
        f"revert (central attention {delta_att:+.1%}, accuracy "
        f"{delta_acc:+.1%}; need attention gain >= +{min_attention_gain:.0%} "
        f"AND accuracy drop <= {max_accuracy_drop:.0%})"
    )


__all__ = [
    "AuditSummary",
    "CAMRowResult",
    "CENTRAL_FRACTION",
    "DEFAULT_NEW_CKPT",
    "DEFAULT_OLD_REPO",
    "audit_engine",
    "keep_or_revert",
    "print_comparison",
    "run_old_vs_new",
]
