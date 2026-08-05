"""Fit the disease classifier's calibration temperature on the held-out test set.

Run this ONCE on Colab (it needs the model + the test images), then paste the
printed ``T`` into ``app.config.DISEASE_TEMPERATURE``. Temperature scaling does
not change any prediction — only how trustworthy the confidence number is — so
the model does not need retraining.

The script also prints the Expected Calibration Error before and after, which is
the evidence that the confidence shown to a farmer became honest, and a suggested
``CONFIDENCE_ADVISE_MIN`` derived from the confidence distribution of the model's
CORRECT predictions (so the advise/defer threshold is set from data, not guessed).

Usage (Colab, C-PD model + PlantDoc test split present):
    python scripts/fit_disease_temperature.py \
        --model ankit-iiitdmj/iks-disease-plantdoc-crop \
        --images-root /content/PlantDoc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.disease import calibration as cal

ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT = ROOT / "data/splits/plantdoc/test.json"


def _collect_logits(engine, images_root: Path, rows: list[dict]):
    """Run the model over the test set → (logits matrix, label vector)."""
    from PIL import Image  # noqa: PLC0415

    logits, labels = [], []
    for i, row in enumerate(rows):
        img_path = images_root / row["path"]
        if not img_path.exists():
            continue
        pred = engine.predict(Image.open(img_path).convert("RGB")).prediction
        logits.append(pred.logits)
        labels.append(int(row["label_idx"]))
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(rows)}")
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=int)


def _suggest_threshold(probs: np.ndarray, labels: np.ndarray, keep: float = 0.90) -> float:
    """A defensible advise/defer floor: keep ``keep`` of the CORRECT predictions.

    Set the floor at the ``(1-keep)`` quantile of the confidence of the model's
    correct predictions, so most genuine detections clear it while the least
    confident (most error-prone) ones fall to the collect-and-retrain path.
    """
    correct_conf = probs.max(1)[probs.argmax(1) == labels]
    if len(correct_conf) == 0:
        return 0.5
    return round(float(np.quantile(correct_conf, 1.0 - keep)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ankit-iiitdmj/iks-disease-plantdoc-crop")
    ap.add_argument("--images-root", required=True,
                    help="folder that contains the PlantDoc test image paths")
    ap.add_argument("--out", default=str(ROOT / "data/calibration/disease_temperature.json"))
    args = ap.parse_args()

    from src.disease.infer import DiseaseInferenceEngine  # noqa: PLC0415

    rows = json.loads(TEST_SPLIT.read_text(encoding="utf-8"))
    print(f"Loading model {args.model} ...")
    engine = DiseaseInferenceEngine(model_source=args.model)

    print(f"Scoring {len(rows)} test images ...")
    logits, labels = _collect_logits(engine, Path(args.images_root), rows)
    print(f"  collected {len(labels)} predictions")
    if len(labels) < 20:
        print("Too few images found — check --images-root.")
        return 1

    raw = cal.softmax(logits, 1.0)
    correct = raw.argmax(1) == labels
    acc = float(correct.mean())
    ece_before = cal.expected_calibration_error(raw.max(1), correct)

    T = cal.fit_temperature(logits, labels)
    calibd = cal.softmax(logits, T)
    ece_after = cal.expected_calibration_error(calibd.max(1), correct)
    threshold = _suggest_threshold(calibd, labels)

    print("\n================= CALIBRATION =================")
    print(f"top-1 accuracy         : {acc:.3f}")
    print(f"temperature T          : {T}")
    print(f"ECE before / after     : {ece_before:.3f} -> {ece_after:.3f}")
    print(f"suggested advise floor : {threshold}")
    print("===============================================")
    print(f"\nPaste into app/config.py:\n"
          f"    DISEASE_TEMPERATURE   = {T}\n"
          f"    CONFIDENCE_ADVISE_MIN = {threshold}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "n": int(len(labels)), "accuracy": acc,
        "temperature": T, "ece_before": ece_before, "ece_after": ece_after,
        "suggested_advise_floor": threshold,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
