"""Smoke tests for the Phase 6 soil-module HF Hub dataset uploads.

These tests verify (a) the three private dataset repos on HF Hub round-
trip cleanly (correct row counts, label ranges, decodable images),
(b) ``configs/data/soil_norm.yaml`` is present and well-formed, and
(c) the multi-task loss-masking helper produces the expected shape.

Network-dependent assertions skip cleanly if HF Hub auth is absent or
the upload has not run yet — that way the test file is safe on a fresh
checkout without credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.soil.dataset import build_multitask_labels
from src.utils.paths import CONFIGS_DIR

PHANTOMFS_CFG_PATH = CONFIGS_DIR / "data" / "soil_soil_type_classes.yaml"
MOISTURE_CFG_PATH = CONFIGS_DIR / "data" / "soil_moisture_classes.yaml"
TEXTURE_CFG_PATH = CONFIGS_DIR / "data" / "soil_texture_classes.yaml"
NORM_PATH = CONFIGS_DIR / "data" / "soil_norm.yaml"

EXPECTED_HF_USERNAME = "ankit-iiitdmj"
REPO_PHANTOMFS = f"{EXPECTED_HF_USERNAME}/iks-soil-phantomfs"
REPO_SIRAJGANJ = f"{EXPECTED_HF_USERNAME}/iks-soil-sirajganj-moisture"
REPO_TEXTURE = f"{EXPECTED_HF_USERNAME}/iks-soil-texture-irsid-vit"

# Expected per-split counts ±tolerance from the prompt's Section D.
EXPECTED_COUNTS = {
    REPO_PHANTOMFS: {"train": (951, 5), "val": (119, 5), "test": (119, 5)},
    REPO_SIRAJGANJ: {"train": (941, 5), "val": (118, 5), "test": (118, 5)},
    REPO_TEXTURE: {"train": (223, 3), "val": (28, 3), "test": (28, 3)},
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hf_api():
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import HfApi
    try:
        api = HfApi()
        api.whoami()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HF Hub auth not available: {exc}")
    return api


@pytest.fixture(scope="module")
def hf_datasets_available():
    pytest.importorskip("datasets")


def _load_split(repo: str, split: str):
    """Load one split from HF Hub, skipping the test if anything is off."""
    pytest.importorskip("datasets")
    from datasets import load_dataset
    try:
        return load_dataset(repo, split=split)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load {repo}/{split}: {exc}")


@pytest.mark.parametrize("repo,head_cfg_path,num_classes", [
    (REPO_PHANTOMFS, PHANTOMFS_CFG_PATH, 7),
    (REPO_SIRAJGANJ, MOISTURE_CFG_PATH, 3),
    (REPO_TEXTURE, TEXTURE_CFG_PATH, 3),
])
def test_split_counts_label_range_and_class_names(repo, head_cfg_path, num_classes, hf_datasets_available):
    cfg = _load_yaml(head_cfg_path)
    expected_class_names = {str(name) for name in cfg["classes"].values()}

    for split in ("train", "val", "test"):
        ds = _load_split(repo, split)
        target, tol = EXPECTED_COUNTS[repo][split]
        assert abs(len(ds) - target) <= tol, (
            f"{repo}/{split} has {len(ds)} rows; expected {target}±{tol}."
        )
        # label_idx is stored as int32/int64; verify in-range and class_name aligned.
        seen_labels = set(ds["label_idx"])
        assert seen_labels.issubset(set(range(num_classes))), (
            f"{repo}/{split} has out-of-range label_idx values: {seen_labels}"
        )
        seen_names = set(ds["class_name"])
        diff = seen_names - expected_class_names
        assert not diff, (
            f"{repo}/{split} has class_name values outside the YAML: {diff}"
        )


@pytest.mark.parametrize("repo", [REPO_PHANTOMFS, REPO_SIRAJGANJ, REPO_TEXTURE])
def test_image_decode_with_pil(repo, hf_datasets_available):
    pytest.importorskip("PIL.Image")
    ds = _load_split(repo, "train")
    # Spot-check the first 3 rows — the prep script verifies all rows pre-push.
    sample = ds.select(range(min(3, len(ds))))
    for row in sample:
        img = row["image"]
        assert img.width > 0 and img.height > 0


def test_hub_has_required_files_and_private_visibility(hf_api):
    for repo in (REPO_PHANTOMFS, REPO_SIRAJGANJ, REPO_TEXTURE):
        try:
            files = hf_api.list_repo_files(repo, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Could not list repo files for {repo}: {exc}")
        # push_to_hub may shard into multiple parquet files under
        # data/<split>-NNNNN-of-MMMMM.parquet, so just ensure each
        # split has at least one parquet file plus a README.
        assert any(f == "README.md" for f in files), f"{repo} missing README; saw {files}"
        for split in ("train", "val", "test"):
            split_files = [f for f in files if split in f and f.endswith(".parquet")]
            assert split_files, f"{repo} has no parquet files for split {split!r}; saw {files}"
        info = hf_api.repo_info(repo, repo_type="dataset")
        assert info.private is True, f"{repo} is not private!"


def test_soil_norm_yaml_is_valid():
    if not NORM_PATH.is_file():
        pytest.skip(f"{NORM_PATH} missing — run scripts/compute_soil_norm_stats.py first.")
    payload = _load_yaml(NORM_PATH)
    assert isinstance(payload["mean"], list) and len(payload["mean"]) == 3
    assert isinstance(payload["std"], list) and len(payload["std"]) == 3
    for m in payload["mean"]:
        assert 0.0 <= float(m) <= 1.0, f"mean out of [0,1]: {m}"
    for s in payload["std"]:
        assert 0.0 < float(s) <= 1.0, f"std out of (0,1]: {s}"
    assert int(payload["n_images"]) > 0


def test_build_multitask_labels_for_soil_type():
    out = build_multitask_labels("soil_type", 3)
    assert out == {"soil_type": 3, "moisture_appearance": -1, "texture": -1}


def test_build_multitask_labels_for_moisture():
    out = build_multitask_labels("moisture_appearance", 1)
    assert out == {"soil_type": -1, "moisture_appearance": 1, "texture": -1}


def test_build_multitask_labels_for_texture():
    out = build_multitask_labels("texture", 2)
    assert out == {"soil_type": -1, "moisture_appearance": -1, "texture": 2}


def test_build_multitask_labels_rejects_unknown_head():
    with pytest.raises(ValueError, match="Unknown head"):
        build_multitask_labels("ph", 0)
