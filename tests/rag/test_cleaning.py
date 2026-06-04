"""Tests for :mod:`src.rag.corpus.cleaning`."""

from __future__ import annotations

import pytest

from src.rag.corpus.cleaning import clean_text, drop_devanagari


def test_drop_devanagari_flags_majority_devanagari_line() -> None:
    # Pure Devanagari script — must drop.
    assert drop_devanagari("अग्नि") is True
    # Mostly Devanagari with one stray Latin letter — still majority.
    assert drop_devanagari("अग्नि x") is True


def test_drop_devanagari_keeps_english_line() -> None:
    assert drop_devanagari("The seed is sown in moist ground.") is False
    # Empty / pure whitespace — also kept (clean_text handles those elsewhere).
    assert drop_devanagari("") is False
    assert drop_devanagari("   ") is False


def test_drop_devanagari_keeps_mixed_majority_english() -> None:
    # Mostly Latin with a Devanagari pronunciation note in parens.
    text = "Apply kunapajala (कुणपजल) to the soil weekly."
    assert drop_devanagari(text) is False


def test_clean_text_removes_standalone_page_number() -> None:
    raw = (
        "Treatment of Trees\n"
        "\n"
        "153\n"
        "\n"
        "Apply paste of butter and clarified ghee.\n"
    )
    cleaned = clean_text(raw)
    assert "153" not in cleaned.splitlines()
    assert "Apply paste of butter" in cleaned
    # Title appears once here (header-denylist may or may not kill it,
    # but the page number line is definitely gone).


def test_clean_text_dehyphenates_line_break_split() -> None:
    raw = (
        "Watering of-\n"
        "trees\n"
        "should be done every fortnight.\n"
    )
    cleaned = clean_text(raw)
    # After dehyphenation the joined word appears once.
    assert "Watering oftrees" in cleaned or "Watering of trees" in cleaned
    assert "should be done every fortnight" in cleaned
    # The orphan "trees" line should not survive on its own.
    assert "\ntrees\n" not in "\n" + cleaned + "\n"


def test_clean_text_drops_devanagari_lines_keeps_english() -> None:
    raw = (
        "अग्नि जलम\n"
        "Translation: The fire of the seed is water.\n"
    )
    cleaned = clean_text(raw)
    assert "Translation" in cleaned
    assert "अ" not in cleaned


def test_clean_text_collapses_whitespace_and_blank_runs() -> None:
    raw = "Line   one\n\n\n\nLine    two\n"
    cleaned = clean_text(raw)
    # Internal whitespace collapsed.
    assert "Line one" in cleaned
    assert "Line two" in cleaned
    # Multiple blank lines collapsed to one paragraph break (single \n\n).
    assert "Line one\n\nLine two" in cleaned


def test_clean_text_filters_running_header() -> None:
    raw = "Brihat Samhita\nApply water to the roots.\n"
    cleaned = clean_text(raw)
    assert "Brihat Samhita" not in cleaned
    assert "Apply water to the roots." in cleaned
