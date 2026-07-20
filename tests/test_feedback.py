"""Feedback-collection tests (app/feedback.py).

Two things matter here:
1. The stored record must carry the farmer's own plant name and must NEVER
   arrive pre-verified — an expert decides before anything becomes training data.
2. Collection is best-effort: no token / no network must not break the advisory.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from PIL import Image

from app import feedback


def _img() -> Image.Image:
    return Image.new("RGB", (32, 32), (10, 120, 60))


# ------------------------------------------------------------------ #
# record shape
# ------------------------------------------------------------------ #


def test_record_keeps_the_farmers_plant_name_and_reason() -> None:
    r = feedback.build_record(
        sample_id="abc123", reason="out_of_scope", declared_plant="brinjal",
    )
    assert r["declared_plant"] == "brinjal"
    assert r["reason"] == "out_of_scope"
    assert r["sample_id"] == "abc123"
    assert r["created_utc"]


def test_new_samples_are_never_pre_verified() -> None:
    """Human-in-the-loop: nothing is training truth until an expert says so."""
    r = feedback.build_record(sample_id="x", reason="low_confidence", declared_plant="tomato")
    assert r["status"] == feedback.STATUS_PENDING
    assert r["expert_label"] is None


def test_record_is_json_serialisable() -> None:
    r = feedback.build_record(
        sample_id="x", reason="low_confidence", declared_plant="tomato",
        predicted_class="Tomato leaf late blight", confidence=0.31, mode="full",
        soil={"soil_type": "Alluvial", "moisture": "moderate", "texture": "mixed"},
    )
    round_tripped = json.loads(json.dumps(r))
    assert round_tripped["confidence"] == pytest.approx(0.31)
    assert round_tripped["soil"]["soil_type"] == "Alluvial"


def test_out_of_scope_samples_have_no_model_prediction() -> None:
    """Out-of-scope short-circuits before inference, so these stay empty."""
    r = feedback.build_record(sample_id="x", reason="out_of_scope", declared_plant="okra")
    assert r["predicted_class"] is None
    assert r["confidence"] is None


# ------------------------------------------------------------------ #
# upload is best-effort
# ------------------------------------------------------------------ #


def test_save_sample_returns_none_when_hub_unavailable(monkeypatch) -> None:
    """No token / no network must NOT raise — the advisory keeps working."""
    broken = types.ModuleType("huggingface_hub")

    class _Api:
        def create_repo(self, **_kw):
            raise RuntimeError("no token")

    broken.HfApi = _Api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", broken)

    assert feedback.save_sample(_img(), reason="out_of_scope", declared_plant="brinjal") is None


def test_save_sample_uploads_image_and_record(monkeypatch) -> None:
    calls: list[dict] = []
    fake = types.ModuleType("huggingface_hub")

    class _Api:
        def create_repo(self, **kw):
            calls.append({"op": "create_repo", **kw})

        def upload_file(self, **kw):
            calls.append({"op": "upload", "path": kw["path_in_repo"],
                          "payload": kw["path_or_fileobj"]})

    fake.HfApi = _Api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    sid = feedback.save_sample(
        _img(), reason="out_of_scope", declared_plant="brinjal", repo_id="acct/test",
    )
    assert sid is not None

    # repo must be created PRIVATE — these are farmers' photos
    created = [c for c in calls if c["op"] == "create_repo"]
    assert created and created[0]["private"] is True

    paths = [c["path"] for c in calls if c["op"] == "upload"]
    assert f"images/{sid}.jpg" in paths
    assert f"records/{sid}.json" in paths

    # one self-contained record per sample => concurrent writes can't clash
    rec = next(c for c in calls if c["op"] == "upload" and c["path"].endswith(".json"))
    parsed = json.loads(rec["payload"].decode("utf-8"))
    assert parsed["declared_plant"] == "brinjal"
    assert parsed["status"] == feedback.STATUS_PENDING
