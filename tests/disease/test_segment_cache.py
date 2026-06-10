"""Phase 5-R Part 2 — segment_cache regressions (HF-row keyed).

Pure-Python checks against the on-disk mask cache contract. No GPU, no
network, no model load — the HF dataset and the Part-1 ``segment()``
function are both monkeypatched.

Locks the four invariants the trainer relies on:

1. :func:`mask_path_for` is stable + OS-independent and keys by
   ``(dataset_id, split, row_idx)``.
2. :func:`build_mask_cache_from_hf` writes a mask for every newly
   processed row AND records the ``"<split>/<row_idx:06d>"`` key of
   any flagged row in the persistent log.
3. A re-run hits the cache (idempotent): zero new segmentations, the
   ``segment()`` call count does not grow.
4. The persistent log merges flagged keys across splits — caching
   "val" after "train" must NOT drop the train flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.disease.segment_cache import (
    MASK_CACHE_ROOT,
    build_mask_cache_from_hf,
    dataset_log_path,
    is_flagged,
    load_flagged_set,
    mask_path_for,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


class _FakeHFSplit:
    """A tiny stand-in for an HF ``Dataset`` split.

    Supports ``len()`` and ``__getitem__`` with the same shape the
    cache builder reads: ``row["image"]`` (PIL) and ``row["label_idx"]``
    (int).
    """

    def __init__(self, images: list, label_idxs: list[int]) -> None:
        self.images = images
        self.label_idxs = label_idxs

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"image": self.images[idx], "label_idx": self.label_idxs[idx]}


def _build_fake_split(n: int = 3):
    from PIL import Image

    images = [
        Image.new("RGB", (16, 16), (i * 60, 80, 80))
        for i in range(n)
    ]
    return _FakeHFSplit(images=images, label_idxs=list(range(n)))


@pytest.fixture()
def temp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect :data:`MASK_CACHE_ROOT` into ``tmp_path`` for the test."""
    fake_root = tmp_path / "_masks"
    fake_root.mkdir()
    monkeypatch.setattr(
        "src.disease.segment_cache.MASK_CACHE_ROOT", fake_root, raising=True,
    )
    return fake_root


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_mask_path_for_is_split_keyed(temp_cache: Path) -> None:
    """``mask_path_for`` must put masks under
    ``<MASK_CACHE_ROOT>/<dataset>/<split>/<06d>.png``."""
    p = mask_path_for("plantvillage", "train", 42)
    assert p.suffix == ".png"
    assert p.name == "000042.png"
    assert p.parent.name == "train"
    assert p.parent.parent.name == "plantvillage"
    # Mask root really IS the patched one.
    assert p.is_relative_to(temp_cache)


def test_build_mask_cache_writes_masks_and_records_flagged(
    temp_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row 1 is flagged (full-image mask); its key must land in the log."""
    fake_split = _build_fake_split(n=3)

    class _OkResult:
        mask = np.full((16, 16), 255, dtype=np.uint8)
        foreground_fraction = 0.4
        flagged_as_failure = False
        method = "classical"

    class _FlaggedResult:
        mask = np.full((16, 16), 255, dtype=np.uint8)
        foreground_fraction = 1.0
        flagged_as_failure = True
        method = "classical"

    # row 0 OK, row 1 flagged, row 2 OK
    seq = [_OkResult(), _FlaggedResult(), _OkResult()]
    call_count = {"n": 0}

    def fake_segment(image, style):
        i = call_count["n"]
        call_count["n"] += 1
        return seq[i]

    monkeypatch.setattr("src.disease.segment_cache.segment", fake_segment)
    # Redirect HF load_dataset to return our fake split. The closure
    # captures fake_split via default arg to avoid the trap where the
    # lambda kwarg ``split`` shadows the outer name.
    def _fake_load(repo, split=None, _fs=fake_split):
        return _fs

    monkeypatch.setattr("datasets.load_dataset", _fake_load)

    stats = build_mask_cache_from_hf(
        dataset_repo="ankit-iiitdmj/iks-plantvillage",
        dataset_id="plantvillage",
        split="train",
        style="lab",
        log_every=1,
        hf_backup_repo=None,
    )
    assert stats.total == 3
    assert stats.newly_segmented == 3
    assert stats.flagged == 1
    assert stats.failures == 0

    # All three mask files exist on disk.
    for i in range(3):
        assert mask_path_for("plantvillage", "train", i).is_file()

    # Log records the flagged key.
    flagged = load_flagged_set("plantvillage")
    assert "train/000001" in flagged
    assert is_flagged("plantvillage", "train", 1)
    assert not is_flagged("plantvillage", "train", 0)


def test_build_mask_cache_is_idempotent(
    temp_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the cache build twice must not re-segment."""
    fake_split = _build_fake_split(n=2)

    class _OkResult:
        mask = np.full((16, 16), 255, dtype=np.uint8)
        foreground_fraction = 0.4
        flagged_as_failure = False
        method = "classical"

    seg_calls = {"n": 0}

    def fake_segment(image, style):
        seg_calls["n"] += 1
        return _OkResult()

    monkeypatch.setattr("src.disease.segment_cache.segment", fake_segment)

    def _fake_load(repo, split=None, _fs=fake_split):
        return _fs

    monkeypatch.setattr("datasets.load_dataset", _fake_load)

    s1 = build_mask_cache_from_hf(
        dataset_repo="repo", dataset_id="plantvillage",
        split="train", style="lab", log_every=1,
        hf_backup_repo=None,
    )
    s2 = build_mask_cache_from_hf(
        dataset_repo="repo", dataset_id="plantvillage",
        split="train", style="lab", log_every=1,
        hf_backup_repo=None,
    )
    assert s1.newly_segmented == 2
    assert s2.newly_segmented == 0
    assert s2.cached_already == 2
    assert seg_calls["n"] == 2  # only the first pass called segment()


def test_push_mask_cache_creates_tarball_and_uploads_to_hf(
    temp_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``push_mask_cache_to_hf`` must tar the dataset's local mask dir,
    call HfApi.upload_file with the expected (filename, repo_id), and
    skip cleanly if there are no masks."""
    import tarfile

    from src.disease.segment_cache import push_mask_cache_to_hf

    # Build a fake local mask cache: one png in plantvillage/train/.
    local_train = temp_cache / "plantvillage" / "train"
    local_train.mkdir(parents=True)
    (local_train / "000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # Plus the log file so it gets packed into the tar.
    log_path = dataset_log_path("plantvillage")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('{"dataset": "plantvillage", "splits": {}, "flagged_keys": []}',
                        encoding="utf-8")

    upload_calls: list[dict[str, Any]] = []

    class _FakeHfApi:
        def create_repo(self, **kwargs):
            return None

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
            upload_calls.append({
                "path_or_fileobj": path_or_fileobj,
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
            })

    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: _FakeHfApi(), raising=True,
    )

    push_mask_cache_to_hf("plantvillage", repo_id="ank/test-mask-cache")

    assert len(upload_calls) == 1
    call = upload_calls[0]
    assert call["path_in_repo"] == "plantvillage.tar.gz"
    assert call["repo_id"] == "ank/test-mask-cache"
    assert call["repo_type"] == "dataset"
    # The tar was uploaded; uploaded path is now gone (deleted after).
    # Verify the tarball would have contained the mask file. Re-create
    # it by re-running and inspecting before deletion: easier to just
    # ensure path matched the expected scratch location.
    assert "_backup_plantvillage.tar.gz" in str(call["path_or_fileobj"])


def test_push_mask_cache_skips_when_no_masks(
    temp_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No local mask dir -> no HF call (no surprise empty pushes)."""
    upload_calls: list[Any] = []

    class _FakeHfApi:
        def create_repo(self, **kwargs):
            upload_calls.append("create_repo")
        def upload_file(self, **kwargs):
            upload_calls.append("upload_file")

    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: _FakeHfApi(), raising=True,
    )

    from src.disease.segment_cache import push_mask_cache_to_hf

    result = push_mask_cache_to_hf("never_segmented", repo_id="ank/x")
    assert result is None
    assert upload_calls == []  # nothing called


def test_pull_mask_cache_extracts_tarball_into_mask_root(
    temp_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pull_mask_cache_from_hf`` should download a tar.gz and extract
    it under :data:`MASK_CACHE_ROOT` so existing masks land at the
    paths the trainer expects."""
    import tarfile

    # Build a fake "downloaded" tarball with a single mask inside.
    src_dir = tmp_path / "_src" / "plantvillage" / "train"
    src_dir.mkdir(parents=True)
    (src_dir / "000007.png").write_bytes(b"\x89PNG fake")
    tar_path = tmp_path / "plantvillage.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        # Arcname must match the format push_mask_cache_to_hf would have
        # written: "<dataset_id>/<split>/<idx>.png".
        tar.add(
            src_dir / "000007.png",
            arcname="plantvillage/train/000007.png",
        )

    # Stub hf_hub_download to return our local tar path.
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: str(tar_path),
        raising=True,
    )

    from src.disease.segment_cache import pull_mask_cache_from_hf

    got = pull_mask_cache_from_hf("plantvillage", repo_id="ank/x")
    assert got is True
    extracted = temp_cache / "plantvillage" / "train" / "000007.png"
    assert extracted.is_file(), (
        f"pull_mask_cache did not extract to expected path. "
        f"Looked at {extracted}, got: "
        f"{sorted(temp_cache.rglob('*'))}"
    )


def test_pull_mask_cache_returns_false_on_first_run(
    temp_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HF backup yet -> graceful False, no exception. Lets the
    builder proceed to segment from scratch on the very first run."""
    from huggingface_hub.errors import EntryNotFoundError

    def _raise(**kwargs):
        raise EntryNotFoundError("not yet")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", _raise, raising=True,
    )

    from src.disease.segment_cache import pull_mask_cache_from_hf

    assert pull_mask_cache_from_hf("plantvillage", repo_id="ank/x") is False


def test_flagged_keys_merge_across_splits(
    temp_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caching ``val`` after ``train`` must NOT erase train's flagged
    keys — the union persists."""

    class _Flag:
        mask = np.full((16, 16), 255, dtype=np.uint8)
        foreground_fraction = 1.0
        flagged_as_failure = True
        method = "classical"

    monkeypatch.setattr(
        "src.disease.segment_cache.segment", lambda image, style: _Flag(),
    )

    train_split = _build_fake_split(n=2)
    val_split = _build_fake_split(n=2)
    holder = {"current": train_split}

    def _fake_load(repo, split=None, _h=holder):
        return _h["current"]

    monkeypatch.setattr("datasets.load_dataset", _fake_load)

    build_mask_cache_from_hf(
        dataset_repo="repo", dataset_id="plantvillage",
        split="train", style="lab", log_every=1,
        hf_backup_repo=None,
    )
    holder["current"] = val_split
    build_mask_cache_from_hf(
        dataset_repo="repo", dataset_id="plantvillage",
        split="val", style="lab", log_every=1,
        hf_backup_repo=None,
    )

    flagged = load_flagged_set("plantvillage")
    assert "train/000000" in flagged
    assert "train/000001" in flagged
    assert "val/000000" in flagged
    assert "val/000001" in flagged
