"""Pick three good Phase 9 demo images from the PlantDoc test set.

A "good" demo image is one where:

1. The Phase 5 disease model predicts the GROUND-TRUTH class (so the
   caption underneath the Grad-CAM matches what we'd want a viewer to
   read — no silent misclassifications like the "Potato → Soyabean
   leaf" case that surprised the first walkthrough).
2. The argmax confidence is ≥ ``CONF_THRESHOLD`` (default 0.85). High
   confidence + correct = the model is sure for the right reason.
3. The Grad-CAM heat-peak falls inside the central 60% of the image —
   not the corners. This filters out the PlantDoc shortcut-bias cases
   where the model attends to a watermark or background instead of the
   leaf.

Output: ranked list of candidates per class, plus a final
recommendation of three samples from distinct classes that we drop
into ``Cell 7`` of the Phase 9 notebook.

Cost: pure local CPU/GPU inference, no API calls. Single-pass over the
256-image test set ≈ 5–10 min on CPU.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.disease.infer import DiseaseInferenceEngine
from src.explain.gradcam import disease_gradcam
from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

_LOGGER = get_logger(__name__)

CONF_THRESHOLD: float = 0.85
CENTRAL_FRACTION: float = 0.60   # CAM peak must fall in central 60x60% box
PLANTDOC_ROOT: Path = PROJECT_ROOT / "data" / "plant_disease" / "plantdoc" / "raw"
TEST_SPLIT: Path = PROJECT_ROOT / "data" / "splits" / "plantdoc" / "test.json"
CACHE_PATH: Path = PROJECT_ROOT / "scripts" / "_phase9_candidates.json"


@dataclass
class Candidate:
    label: str
    rel_path: str
    abs_path: Path
    pred_label: str
    pred_conf: float
    pred_index: int
    peak_y: float    # in [0, 1] image-relative
    peak_x: float
    correct: bool
    central: bool

    def score(self) -> float:
        """Sort key — higher is better. Confidence + central-peak bonus."""
        bonus = 0.05 if self.central else 0.0
        return float(self.pred_conf) + bonus


def _load_test_split() -> list[dict[str, Any]]:
    with TEST_SPLIT.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _peak_position(heatmap: Any) -> tuple[float, float]:
    """Return the (y, x) of the CAM argmax in [0, 1] image-relative coords."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(heatmap, dtype=float)
    flat_idx = int(np.argmax(arr))
    h, w = arr.shape[:2]
    y, x = divmod(flat_idx, w)
    return (float(y) / max(1, h - 1), float(x) / max(1, w - 1))


def _is_central(peak_y: float, peak_x: float, fraction: float = CENTRAL_FRACTION) -> bool:
    half = fraction / 2.0
    return (0.5 - half) <= peak_y <= (0.5 + half) and (0.5 - half) <= peak_x <= (0.5 + half)


def _score_all(samples: list[dict[str, Any]]) -> list[Candidate]:
    """Score every test sample. Result is also cached to CACHE_PATH so a
    print/format bug later in main() never re-burns the 9-min compute."""
    engine = DiseaseInferenceEngine(
        model_source="ankit-iiitdmj/iks-disease-plantdoc",
        device="cpu",
    )

    candidates: list[Candidate] = []
    skipped = 0
    for i, sample in enumerate(samples, start=1):
        rel = sample["path"]
        label = sample["label"]
        abs_path = PLANTDOC_ROOT / rel
        if not abs_path.is_file():
            skipped += 1
            continue
        try:
            cam = disease_gradcam(abs_path, engine)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("CAM failed for %s: %s", rel, exc)
            skipped += 1
            continue

        peak_y, peak_x = _peak_position(cam.heatmap)
        candidates.append(Candidate(
            label=label,
            rel_path=rel,
            abs_path=abs_path,
            pred_label=cam.pred_label,
            pred_conf=cam.pred_conf,
            pred_index=cam.pred_index,
            peak_y=peak_y,
            peak_x=peak_x,
            correct=(cam.pred_label == label),
            central=_is_central(peak_y, peak_x),
        ))
        if i % 25 == 0 or i == len(samples):
            qual = sum(
                1 for c in candidates
                if c.correct and c.pred_conf >= CONF_THRESHOLD and c.central
            )
            print(f"  scored {i}/{len(samples)}  (skipped={skipped}, qualified so far={qual})")

    # Save raw candidates so the print step is independent of the
    # scoring step from now on.
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            [{
                "label": c.label, "rel_path": c.rel_path,
                "pred_label": c.pred_label, "pred_conf": c.pred_conf,
                "pred_index": c.pred_index,
                "peak_y": c.peak_y, "peak_x": c.peak_x,
                "correct": c.correct, "central": c.central,
            } for c in candidates],
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nCached {len(candidates)} candidates to {CACHE_PATH.relative_to(PROJECT_ROOT)}")
    return candidates


def _load_cached() -> list[Candidate] | None:
    if not CACHE_PATH.is_file():
        return None
    raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return [
        Candidate(
            label=r["label"], rel_path=r["rel_path"],
            abs_path=PLANTDOC_ROOT / r["rel_path"],
            pred_label=r["pred_label"], pred_conf=r["pred_conf"],
            pred_index=r["pred_index"],
            peak_y=r["peak_y"], peak_x=r["peak_x"],
            correct=r["correct"], central=r["central"],
        )
        for r in raw
    ]


def main() -> int:
    print(f"PlantDoc test set: {TEST_SPLIT.relative_to(PROJECT_ROOT)}")
    samples = _load_test_split()
    print(f"  {len(samples)} samples\n")

    candidates = _load_cached()
    if candidates is not None:
        print(f"Using cached scores from {CACHE_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(candidates)} samples). Delete the file to force re-score.\n")
    else:
        candidates = _score_all(samples)

    print()
    print("=" * 78)
    print(f"Done. {len(candidates)} scored.")

    # Filter breakdown -- this is the diagnostic the user needs to know
    # whether the Phase 9 "blank-heatmap" surprise is a model-accuracy
    # issue or a model-attention issue.
    n_correct = sum(1 for c in candidates if c.correct)
    n_high_conf = sum(1 for c in candidates if c.pred_conf >= CONF_THRESHOLD)
    n_central = sum(1 for c in candidates if c.central)
    n_correct_high = sum(
        1 for c in candidates if c.correct and c.pred_conf >= CONF_THRESHOLD
    )
    n_correct_central = sum(
        1 for c in candidates if c.correct and c.central
    )
    qualified = [
        c for c in candidates
        if c.correct and c.pred_conf >= CONF_THRESHOLD and c.central
    ]

    total = len(candidates)
    print()
    print("=== Breakdown (per-filter counts over {} test images) ===".format(total))
    print(f"  correct prediction (matches GT label) : {n_correct:>3} / {total} "
          f"({n_correct/total:>6.1%})")
    print(f"  high confidence (>= {CONF_THRESHOLD:.2f})            : {n_high_conf:>3} / {total} "
          f"({n_high_conf/total:>6.1%})")
    print(f"  CAM peak in central {int(CENTRAL_FRACTION*100)}% box           : {n_central:>3} / {total} "
          f"({n_central/total:>6.1%})")
    print(f"  correct AND high-conf                 : {n_correct_high:>3} / {total} "
          f"({n_correct_high/total:>6.1%})")
    print(f"  correct AND central                   : {n_correct_central:>3} / {total} "
          f"({n_correct_central/total:>6.1%})")
    print(f"  ALL three (qualified)                 : {len(qualified):>3} / {total} "
          f"({len(qualified)/total:>6.1%})")
    print()
    print("What this tells us:")
    if n_correct / total >= 0.7:
        print(f"  - Model accuracy on this test slice is reasonable "
              f"({n_correct/total:.0%}). The Phase 5 numbers were real.")
    else:
        print(f"  - Model accuracy on this test slice is LOW "
              f"({n_correct/total:.0%}). The Phase 5 numbers may need re-checking.")
    if n_central / total < 0.3:
        print(f"  - CAM peak lands in the central 60% only "
              f"{n_central/total:.0%} of the time. The model is paying lots of")
        print("    attention to image corners / borders -- classic shortcut")
        print("    bias on PlantDoc (watermarks, backgrounds, framing).")
    elif n_central / total < 0.6:
        print(f"  - CAM peak lands in the central 60% box "
              f"{n_central/total:.0%} of the time. Mixed attention -- some")
        print("    samples are clean, others show shortcut bias.")
    else:
        print(f"  - CAM peak lands in the central 60% box "
              f"{n_central/total:.0%} of the time. Attention is reasonably on-leaf.")
    print()

    # Group by class so we report one row per available disease class.
    by_class: dict[str, list[Candidate]] = defaultdict(list)
    for c in qualified:
        by_class[c.label].append(c)

    print()
    print(f"=== Best candidate per class (strict qualified, n={len(qualified)}) ===")
    if not by_class:
        print("  (none qualified under the strict filter; relaxing...)")
    print(f"{'label':<32} {'conf':>5} {'peak(y,x)':<12} {'file':<48}")
    print("-" * 105)
    per_class_best: list[Candidate] = []
    for label in sorted(by_class):
        rows = sorted(by_class[label], key=lambda r: -r.score())
        best = rows[0]
        per_class_best.append(best)
        print(
            f"{label:<32} {best.pred_conf:>5.2f} "
            f"({best.peak_y:.2f},{best.peak_x:.2f}) "
            f"{best.rel_path[:46]}"
        )

    # If the strict filter found < 3 distinct-class candidates, relax
    # to "correct + high-conf" (drop the central-peak constraint) and
    # surface those — they still produce honest demo panels even though
    # the attention may be near a border.
    if len(per_class_best) < 3:
        print()
        print("=== Strict filter under-supplied; relaxing to correct+high-conf only ===")
        relaxed = [c for c in candidates if c.correct and c.pred_conf >= CONF_THRESHOLD]
        relaxed_by_class: dict[str, list[Candidate]] = defaultdict(list)
        for c in relaxed:
            relaxed_by_class[c.label].append(c)
        print(f"{'label':<32} {'conf':>5} {'peak(y,x)':<12} {'central':<8} {'file':<40}")
        print("-" * 115)
        for label in sorted(relaxed_by_class):
            rows = sorted(relaxed_by_class[label], key=lambda r: -r.pred_conf)
            best = rows[0]
            print(
                f"{label:<32} {best.pred_conf:>5.2f} "
                f"({best.peak_y:.2f},{best.peak_x:.2f}) "
                f"{'yes' if best.central else 'no':<8} "
                f"{best.rel_path[:40]}"
            )
            if best not in per_class_best:
                per_class_best.append(best)

    # Pick 3 distinct-class samples maximising score.
    per_class_best.sort(key=lambda c: -c.score())
    top3 = per_class_best[:3]
    print()
    print("=" * 78)
    print("RECOMMENDATION — three demo samples for Cell 7:")
    print("=" * 78)
    for c in top3:
        print(f"  label : {c.label}")
        print(f"  file  : {c.rel_path}")
        print(f"  conf  : {c.pred_conf:.3f}    peak: ({c.peak_y:.2f}, {c.peak_x:.2f})")
        print()

    # Emit ready-to-paste PLANTDOC_TARGETS dict for the notebook.
    print("=" * 78)
    print("Ready-to-paste PLANTDOC_TARGETS for build_phase9_notebook.py / Cell 7:")
    print("=" * 78)
    print("PLANTDOC_TARGETS = {")
    crops = ["leaf1", "leaf2", "leaf3"]
    for crop_key, c in zip(crops, top3, strict=True):
        rel_disp = c.rel_path.replace("\\", "/")
        print(f"    {crop_key!r}: (")
        print(f"        {c.label!r},")
        print(f"        {rel_disp!r},")
        print(f"    ),")
    print("}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
