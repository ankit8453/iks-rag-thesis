"""Collect real-world samples the system could not handle, for later retraining.

Why this exists
---------------
The classifier knows 13 plants. When a farmer submits something outside that set
(or the model is very unsure), throwing the image away wastes exactly the data
that would let the model grow. We keep it instead — but *only* as a candidate:
nothing here is treated as training truth until a domain expert confirms the
label. That human-in-the-loop step is deliberate; learning directly from
unverified user input would poison the model.

Where it goes
-------------
A private Hugging Face dataset repo (:data:`app.config.FEEDBACK_REPO`). Colab's
local disk is wiped when the session ends, so writing there would silently lose
every sample. HF persists, is already authenticated in our notebooks, and can be
shared with the agronomy expert for review.

Layout — one self-contained record per sample, never a shared index file, so
concurrent submissions can't clash or overwrite each other::

    images/<sample_id>.jpg     the leaf photo
    records/<sample_id>.json   what we knew at submission time

Failure to upload is never fatal: the advisory must keep working even with no
token or no network.
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app import config as app_config
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

#: Status every new sample starts in — an expert must review before it is
#: eligible to become training data.
STATUS_PENDING: str = "pending_expert_review"


def build_record(
    *,
    sample_id: str,
    reason: str,
    declared_plant: str,
    predicted_class: str | None = None,
    confidence: float | None = None,
    mode: str | None = None,
    soil: dict[str, str] | None = None,
    notes: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble the metadata stored alongside a collected image.

    Kept as a pure function so the record shape is testable without network,
    a token, or a live model.
    """
    return {
        "sample_id": sample_id,
        "created_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        # why we kept this sample: "out_of_scope" | "low_confidence"
        "reason": reason,
        # what the FARMER said it is — for an out-of-scope plant this is the
        # only label we have, and it is what makes the sample useful later.
        "declared_plant": declared_plant,
        # what the model thought, if it was run at all (out-of-scope samples
        # short-circuit before inference, so these stay None).
        "predicted_class": predicted_class,
        "confidence": confidence,
        "mode": mode,
        "soil": soil or {},
        "notes": notes or "",
        # never "verified" on arrival — an expert decides.
        "status": STATUS_PENDING,
        "expert_label": None,
    }


def _encode_jpeg(image: Any) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def save_sample(
    image: Any,
    *,
    reason: str,
    declared_plant: str,
    predicted_class: str | None = None,
    confidence: float | None = None,
    mode: str | None = None,
    soil: dict[str, str] | None = None,
    notes: str | None = None,
    repo_id: str | None = None,
) -> str | None:
    """Upload one sample (image + record) to the feedback dataset.

    Returns the ``sample_id`` on success, or ``None`` if the sample could not be
    stored — callers should treat ``None`` as "not collected" and carry on. This
    never raises: a failed upload must not break the farmer's advisory.
    """
    repo = repo_id or app_config.FEEDBACK_REPO
    sample_id = uuid.uuid4().hex[:16]
    record = build_record(
        sample_id=sample_id, reason=reason, declared_plant=declared_plant,
        predicted_class=predicted_class, confidence=confidence, mode=mode,
        soil=soil, notes=notes,
    )

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi()
        # create_repo is idempotent; private so farmer photos are never public.
        api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=_encode_jpeg(image),
            path_in_repo=f"images/{sample_id}.jpg",
            repo_id=repo, repo_type="dataset",
            commit_message=f"feedback sample {sample_id} ({reason})",
        )
        api.upload_file(
            path_or_fileobj=json.dumps(record, indent=2).encode("utf-8"),
            path_in_repo=f"records/{sample_id}.json",
            repo_id=repo, repo_type="dataset",
            commit_message=f"feedback record {sample_id}",
        )
    except Exception as exc:  # noqa: BLE001 - collection is best-effort by design
        _LOGGER.warning("Feedback sample not stored (%s): %s", reason, exc)
        return None

    _LOGGER.info("Stored feedback sample %s (%s) in %s", sample_id, reason, repo)
    return sample_id


__all__ = ["STATUS_PENDING", "build_record", "save_sample"]
