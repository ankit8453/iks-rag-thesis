"""Tests for src.utils.hf_upload.

Real HF Hub access is mocked — we never push during pytest. The tests
verify our helper makes the right API calls in the right order with
the right arguments.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Sentinel fixtures kept top-level so each test can mint a fresh fake
# dataset on disk in `tmp_path`.

def _make_fake_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a tiny on-disk fake matching the Phase 4 split layout."""
    pil = pytest.importorskip("PIL.Image")
    local_root = tmp_path / "raw"
    (local_root / "a").mkdir(parents=True)
    (local_root / "b").mkdir(parents=True)
    for label in ("a", "b"):
        for i in range(2):
            pil.new("RGB", (32, 32), color=(i * 40, 40, 60)).save(
                local_root / label / f"{i}.jpg", format="JPEG"
            )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split_name, rows in {
        "train": [
            {"path": "a/0.jpg", "label": "a", "label_idx": 0},
            {"path": "b/0.jpg", "label": "b", "label_idx": 1},
        ],
        "val": [{"path": "a/1.jpg", "label": "a", "label_idx": 0}],
        "test": [{"path": "b/1.jpg", "label": "b", "label_idx": 1}],
    }.items():
        (splits_dir / f"{split_name}.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )

    class_map_path = splits_dir / "class_map.json"
    class_map_path.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
    return local_root, splits_dir, class_map_path


# ----------------------------------------------------------------------
# Pre-flight: whoami / token role gating
# ----------------------------------------------------------------------


def test_constructor_rejects_wrong_username() -> None:
    pytest.importorskip("huggingface_hub")
    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api = MagicMock()
        fake_api.whoami.return_value = {
            "name": "someone-else",
            "auth": {"accessToken": {"role": "write"}},
        }
        fake_api_cls.return_value = fake_api
        from src.utils.hf_upload import HFDatasetUploader

        with pytest.raises(PermissionError, match="someone-else"):
            HFDatasetUploader(expected_username="ankit-iiitdmj")


def test_constructor_rejects_read_only_token() -> None:
    pytest.importorskip("huggingface_hub")
    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api = MagicMock()
        fake_api.whoami.return_value = {
            "name": "ankit-iiitdmj",
            "auth": {"accessToken": {"role": "read"}},
        }
        fake_api_cls.return_value = fake_api
        from src.utils.hf_upload import HFDatasetUploader

        with pytest.raises(PermissionError, match="role is 'read'"):
            HFDatasetUploader(expected_username="ankit-iiitdmj")


def test_constructor_accepts_matching_write_user() -> None:
    pytest.importorskip("huggingface_hub")
    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api = MagicMock()
        fake_api.whoami.return_value = {
            "name": "ankit-iiitdmj",
            "auth": {"accessToken": {"role": "write"}},
        }
        fake_api_cls.return_value = fake_api
        from src.utils.hf_upload import HFDatasetUploader

        uploader = HFDatasetUploader(expected_username="ankit-iiitdmj")
        assert uploader is not None
        fake_api.whoami.assert_called_once()


# ----------------------------------------------------------------------
# Upload flow: right API calls, right order
# ----------------------------------------------------------------------


def _good_api() -> MagicMock:
    fake_api = MagicMock()
    fake_api.whoami.return_value = {
        "name": "ankit-iiitdmj",
        "auth": {"accessToken": {"role": "write"}},
    }
    return fake_api


def test_upload_calls_create_repo_and_push_to_hub(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    datasets_mod = pytest.importorskip("datasets")
    local_root, splits_dir, class_map_path = _make_fake_dataset(tmp_path)

    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api_cls.return_value = _good_api()
        from src.utils.hf_upload import HFDatasetUploader

        uploader = HFDatasetUploader()

        # Stub out DatasetDict.push_to_hub so no network happens.
        with patch.object(
            datasets_mod.DatasetDict, "push_to_hub", return_value=None
        ) as push_mock:
            result = uploader.upload_image_classification_dataset(
                local_root=local_root,
                splits_dir=splits_dir,
                class_map_path=class_map_path,
                hub_repo_id="ankit-iiitdmj/iks-fixture",
                private=True,
                license_str="cc0-1.0",
                dataset_card_body="Tiny fixture dataset for unit tests.",
            )

        # create_repo called once on the dataset repo.
        assert any(
            call.kwargs.get("repo_id") == "ankit-iiitdmj/iks-fixture"
            and call.kwargs.get("repo_type") == "dataset"
            and call.kwargs.get("private") is True
            for call in uploader._api.create_repo.call_args_list
        ), "create_repo not called with the right args"

        push_mock.assert_called_once_with("ankit-iiitdmj/iks-fixture", private=True)

        # README upload happened.
        upload_calls = uploader._api.upload_file.call_args_list
        readmes = [
            c for c in upload_calls if c.kwargs.get("path_in_repo") == "README.md"
        ]
        assert len(readmes) == 1, "Dataset card README.md not uploaded exactly once"

        assert result.n_train == 2
        assert result.n_val == 1
        assert result.n_test == 1
        assert result.n_classes == 2
        assert result.hub_url == "https://huggingface.co/datasets/ankit-iiitdmj/iks-fixture"


def test_load_split_entries_raises_on_missing_image(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    local_root, splits_dir, class_map_path = _make_fake_dataset(tmp_path)

    # Stomp on one of the images so the resolution fails.
    (local_root / "a" / "0.jpg").unlink()

    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api_cls.return_value = _good_api()
        from src.utils.hf_upload import HFDatasetUploader

        uploader = HFDatasetUploader()
        with pytest.raises(FileNotFoundError, match="a/0.jpg"):
            uploader._load_split_entries(splits_dir, local_root)


def test_dataset_card_renders_with_classes_and_sizes(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    local_root, splits_dir, class_map_path = _make_fake_dataset(tmp_path)

    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api_cls.return_value = _good_api()
        from src.utils.hf_upload import HFDatasetUploader

        uploader = HFDatasetUploader()

        # Capture the README payload by stubbing upload_file.
        captured: dict[str, bytes] = {}

        def _capture(*args, **kwargs):
            if kwargs.get("path_in_repo") == "README.md":
                captured["body"] = kwargs["path_or_fileobj"]
            return MagicMock()

        with patch.object(uploader._api, "upload_file", side_effect=_capture):
            uploader._write_dataset_card(
                hub_repo_id="ankit-iiitdmj/iks-fixture",
                class_map={"a": 0, "b": 1},
                n_train=2,
                n_val=1,
                n_test=1,
                license_str="cc0-1.0",
                body="Tiny dataset.",
            )

        body = captured["body"].decode("utf-8")
        assert "license: cc0-1.0" in body
        assert "# ankit-iiitdmj/iks-fixture" in body
        assert "Tiny dataset." in body
        assert "- `a` (idx 0)" in body
        assert "- `b` (idx 1)" in body
        assert "train: 2" in body
        assert "val: 1" in body
        assert "test: 1" in body
        assert "total: 4" in body
        assert "classes: 2" in body
