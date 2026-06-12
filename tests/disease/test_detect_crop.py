"""Tests for the detect-then-crop helpers (VOC parse, crop, class match).

CPU-only, no network, no model. Guards the small logic the PlantDoc
detect-then-crop pipeline depends on so a regression bites here, not in
a 30-minute Colab run.
"""

from __future__ import annotations

import textwrap

import pytest
from PIL import Image

from src.disease.detect_crop import (
    Box,
    build_normalized_class_map,
    crop_to_box,
    match_class_to_index,
    normalize_class_name,
    parse_voc_xml,
)


def _write_xml(tmp_path, body: str):
    p = tmp_path / "ann.xml"
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_voc_xml_reads_boxes(tmp_path) -> None:
    xml = textwrap.dedent("""\
        <annotation>
          <filename>img1.jpg</filename>
          <size><width>400</width><height>300</height></size>
          <object>
            <name>Apple Scab leaf</name>
            <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>220</ymax></bndbox>
          </object>
          <object>
            <name>Apple leaf</name>
            <bndbox><xmin>200</xmin><ymin>50</ymin><xmax>380</xmax><ymax>250</ymax></bndbox>
          </object>
        </annotation>
    """)
    boxes = parse_voc_xml(_write_xml(tmp_path, xml))
    assert len(boxes) == 2
    assert boxes[0].name == "Apple Scab leaf"
    assert (boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax) == (10, 20, 110, 220)
    assert boxes[0].width == 100 and boxes[0].height == 200


def test_parse_voc_xml_drops_degenerate_box(tmp_path) -> None:
    """Zero-area / malformed boxes must be dropped, not returned."""
    xml = textwrap.dedent("""\
        <annotation>
          <object>
            <name>good</name>
            <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
          </object>
          <object>
            <name>zero area</name>
            <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
          </object>
        </annotation>
    """)
    boxes = parse_voc_xml(_write_xml(tmp_path, xml))
    assert len(boxes) == 1
    assert boxes[0].name == "good"


def test_crop_to_box_dimensions() -> None:
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    box = Box("x", 10, 20, 110, 220)
    crop = crop_to_box(img, box)
    assert crop.size == (100, 200)


def test_crop_to_box_padding_clamped() -> None:
    """Padding expands the box but stays within image bounds."""
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    box = Box("x", 5, 5, 105, 105)  # 100x100
    crop = crop_to_box(img, box, pad_frac=0.5)  # +50px each side -> clamps at 0
    # left/top clamp to 0; right/bottom expand to 155
    assert crop.size == (155, 155)


def test_crop_to_box_degenerate_falls_back_to_full() -> None:
    img = Image.new("RGB", (50, 50))
    box = Box("x", 30, 30, 10, 10)  # inverted -> degenerate
    crop = crop_to_box(img, box)
    assert crop.size == (50, 50)  # whole image, no crash


def test_normalize_class_name() -> None:
    assert normalize_class_name("Bell_pepper leaf  spot") == "bell pepper leaf spot"
    assert normalize_class_name("Apple Scab Leaf") == "apple scab leaf"
    assert normalize_class_name("  grape leaf black rot ") == "grape leaf black rot"


def test_match_class_handles_cosmetic_differences() -> None:
    """The classifier map and the XML use cosmetically different labels;
    matching must still succeed (and miss cleanly on a true unknown)."""
    class_map = {"Apple Scab Leaf": 0, "Bell_pepper leaf spot": 4, "grape leaf": 25}
    norm = build_normalized_class_map(class_map)
    assert match_class_to_index("Apple Scab leaf", norm) == 0       # case differs
    assert match_class_to_index("Bell pepper leaf spot", norm) == 4  # _ vs space
    assert match_class_to_index("grape leaf", norm) == 25
    assert match_class_to_index("Totally Unknown Class", norm) is None
