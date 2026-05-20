"""Tests for src.utils.logging_setup."""

from __future__ import annotations

import logging

from src.utils.logging_setup import get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("iks.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "iks.test"


def test_root_logger_has_at_least_one_handler() -> None:
    get_logger("iks.handler_check")
    root = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
