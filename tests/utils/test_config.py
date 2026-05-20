"""Tests for src.utils.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config import BaseConfig, dump_config, load_config


class ExampleConfig(BaseConfig):
    name: str
    epochs: int = 10


def test_extra_keys_are_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "extra.yaml"
    cfg_path.write_text("name: foo\nepochs: 3\nmystery_key: nope\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(cfg_path, ExampleConfig)


def test_missing_required_keys_are_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "missing.yaml"
    cfg_path.write_text("epochs: 3\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(cfg_path, ExampleConfig)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml", ExampleConfig)


def test_round_trip_dump_then_load(tmp_path: Path) -> None:
    cfg = ExampleConfig(name="round_trip", epochs=5)
    out = tmp_path / "dumped.yaml"
    dump_config(cfg, out)

    loaded = load_config(out, ExampleConfig)
    assert loaded == cfg


def test_config_is_frozen() -> None:
    cfg = ExampleConfig(name="immutable")
    with pytest.raises(ValidationError):
        cfg.name = "changed"  # type: ignore[misc]
