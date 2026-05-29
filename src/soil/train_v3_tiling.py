"""Phase 6 V3-tiling — V2-recipe training driver on the tiled texture data.

V3-tiling is a **single-stage** experiment that runs the V2 training
recipe **unchanged** on a different texture dataset (the 4× more
patches in ``ankit-iiitdmj/iks-soil-texture-tiled``). The only new code
here is:

- :class:`SoilCheckpointManagerV3Tiling` — subclass of V2's checkpoint
  manager pinned to the new model repo
  ``ankit-iiitdmj/iks-soil-multitask-v3-tiling``.
- :func:`health_check` — abort after epoch 1 if any head's val top-1 is
  below 40%. This is the explicit guard against the catastrophic-collapse
  failure mode from the abandoned 3-stage V3 experiment, where all heads
  collapsed to ~20% accuracy.

The 3-stage sequential-transfer pattern from the earlier V3 attempt is
**BANNED** (see PHASE6_V3_TILING_PROMPT.md). This module does NOT
implement any staged or curriculum schedule — it only re-exports V2
training functions for the notebook to compose.
"""

from __future__ import annotations

from typing import Any

from src.soil.mixup import (  # noqa: F401  re-exports for the notebook
    cutmix_data,
    maybe_apply_mix,
    mixed_loss,
    mixup_data,
)
from src.soil.model import SoilMultiTaskClassifier  # noqa: F401
from src.soil.train import TASK_WEIGHTS, evaluate_per_task  # noqa: F401
from src.soil.train_v2 import (  # noqa: F401
    SoilCheckpointManagerV2,
    compute_multitask_loss_smoothed,
    evaluate_per_task_tta,
    train_one_epoch_v2,
)
from src.soil.transforms import build_soil_eval_aug  # noqa: F401
from src.soil.transforms_v2 import build_soil_train_aug_v2, build_tta_views  # noqa: F401
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


DEFAULT_MODEL_REPO_V3_TILING: str = "ankit-iiitdmj/iks-soil-multitask-v3-tiling"

# Source HF Hub repos used at training time. Texture is the only one
# that changes from V2: it points at the tiled patch dataset.
HF_DATASETS_V3_TILING: dict[str, str] = {
    "soil_type": "ankit-iiitdmj/iks-soil-phantomfs",
    "moisture":  "ankit-iiitdmj/iks-soil-sirajganj-moisture",
    "texture":   "ankit-iiitdmj/iks-soil-texture-tiled",
}

# Below this top-1 on the FIRST epoch's val pass we abort: the V2 recipe
# already reaches ~70-85% on at least one head within one frozen-backbone
# epoch, so anything well under 40% means something is structurally
# broken — better to stop than burn 10 more hours.
HEALTH_THRESHOLD: float = 0.40


class SoilCheckpointManagerV3Tiling(SoilCheckpointManagerV2):
    """Same contract as V2's manager, pinned to the V3-tiling model repo.

    Inherits ``save`` / ``save_best`` / ``try_load_latest`` /
    ``try_load_best`` unchanged so the V3-tiling notebook reads
    naturally side-by-side with V2.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_MODEL_REPO_V3_TILING,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(repo_id=repo_id, *args, **kwargs)


def health_check(
    val_metrics: dict[str, dict[str, float]],
    threshold: float = HEALTH_THRESHOLD,
) -> None:
    """Abort if any head's val top-1 is below ``threshold``.

    Intended for the FIRST epoch's val pass — if a head is still under
    40% by then, the run is almost certainly headed for collapse and
    the remaining ~10 hours of Colab time should NOT be burned. The
    notebook calls this inside Cell 9's training loop right after the
    first ``evaluate_per_task`` finishes.

    Parameters
    ----------
    val_metrics
        ``{head: {"top1_accuracy": float, "macro_f1": float, "n_samples": int}}``
        — output of :func:`src.soil.train.evaluate_per_task`.
    threshold
        Minimum acceptable top-1 per head. Default 0.40.

    Raises
    ------
    RuntimeError
        If any head's top-1 is strictly below ``threshold``. The error
        message names every failing head and its measured top-1.
    """
    bad: list[tuple[str, float]] = []
    for head, m in val_metrics.items():
        top1 = float(m.get("top1_accuracy", 0.0))
        if top1 < threshold:
            bad.append((head, top1))

    if bad:
        details = ", ".join(f"{h} top-1={t:.4f}" for h, t in bad)
        raise RuntimeError(
            f"Epoch-1 health check FAILED — heads below threshold "
            f"({threshold:.2f}): {details}. The run is heading for collapse; "
            f"abort BEFORE burning more Colab time. Investigate before retrying."
        )
    _LOGGER.info(
        "Epoch-1 health check ok (threshold=%.2f): %s",
        threshold,
        ", ".join(f"{h}={float(m['top1_accuracy']):.4f}" for h, m in val_metrics.items()),
    )


__all__ = [
    "DEFAULT_MODEL_REPO_V3_TILING",
    "HF_DATASETS_V3_TILING",
    "HEALTH_THRESHOLD",
    "SoilCheckpointManagerV3Tiling",
    "health_check",
]
