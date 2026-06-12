"""Detect-then-crop helpers for the PlantDoc disease fix (Step 2).

Phase 5 audit (see EXPERIMENT_LOG.md) showed the disease model attends to
background, not the leaf, on cluttered PlantDoc images — and that two
training-side fixes (background randomization, LP-FT) both failed. The
literature's strongest lever is **crop the leaf, then classify**: the
original PlantDoc paper reports uncropped 29.7% -> ground-truth-cropped
70.5%.

This module holds the small, testable pieces of that pipeline:

- parse PlantDoc's VOC-XML ground-truth boxes,
- crop a PIL image to a box (with optional padding),
- map a box's class name to the classifier's class-index.

The heavy parts (running EfficientNet-B4, running a YOLO leaf detector)
stay in the notebook — they need a GPU + downloaded weights. These
helpers are pure-Python + Pillow so they unit-test on CPU with no
network.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class Box:
    """One ground-truth bounding box from a VOC XML annotation.

    Coordinates are pixel ints, ``[xmin, ymin, xmax, ymax]`` with the
    origin at the top-left (VOC convention).
    """

    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)


def parse_voc_xml(xml_path: str | Path) -> list[Box]:
    """Parse a PlantDoc VOC-XML annotation into a list of :class:`Box`.

    PlantDoc detection annotations follow the Pascal-VOC schema::

        <annotation>
          <object>
            <name>Apple Scab leaf</name>
            <bndbox><xmin>..</xmin><ymin>..</ymin>
                    <xmax>..</xmax><ymax>..</ymax></bndbox>
          </object>
          ...
        </annotation>

    Boxes with a non-positive area are dropped (a handful of PlantDoc
    annotations are malformed). Returns ``[]`` for an annotation with no
    valid objects.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    boxes: list[Box] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if name_el is None or bb is None or not (name_el.text or "").strip():
            continue
        try:
            xmin = int(float(bb.findtext("xmin", "0")))
            ymin = int(float(bb.findtext("ymin", "0")))
            xmax = int(float(bb.findtext("xmax", "0")))
            ymax = int(float(bb.findtext("ymax", "0")))
        except (TypeError, ValueError):
            continue
        box = Box(name=name_el.text.strip(), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    return boxes


def crop_to_box(image: Any, box: Box, pad_frac: float = 0.0) -> Any:
    """Crop a PIL image to ``box``, optionally padding by ``pad_frac``.

    ``pad_frac`` expands the box on every side by that fraction of the
    box's width/height (e.g. ``0.1`` = 10% margin) so a tight box doesn't
    clip the leaf edge. The padded box is clamped to the image bounds.
    Returns a new PIL image (RGB).
    """
    w, h = image.size
    px = int(round(box.width * pad_frac))
    py = int(round(box.height * pad_frac))
    left = max(0, box.xmin - px)
    top = max(0, box.ymin - py)
    right = min(w, box.xmax + px)
    bottom = min(h, box.ymax + py)
    if right <= left or bottom <= top:
        # Degenerate box — fall back to the whole image rather than crash.
        return image.convert("RGB") if image.mode != "RGB" else image
    crop = image.crop((left, top, right, bottom))
    return crop.convert("RGB") if crop.mode != "RGB" else crop


def normalize_class_name(name: str) -> str:
    """Canonicalise a class name for matching across datasets.

    PlantDoc's detection XML and the classifier's ``class_map.json`` use
    the same labels but with cosmetic differences (case, ``_`` vs space,
    double spaces). Lower-case, turn ``_`` into spaces, collapse runs of
    whitespace, strip. So ``"Bell_pepper leaf  spot"`` and
    ``"bell pepper leaf spot"`` both map to ``"bell pepper leaf spot"``.
    """
    s = name.replace("_", " ").lower().strip()
    return re.sub(r"\s+", " ", s)


def build_normalized_class_map(class_map: dict[str, int]) -> dict[str, int]:
    """Return ``{normalized_name: index}`` from a ``{name: index}`` map."""
    return {normalize_class_name(k): int(v) for k, v in class_map.items()}


def match_class_to_index(
    name: str, normalized_class_map: dict[str, int],
) -> int | None:
    """Map a box's class name to a classifier index, or ``None`` if absent.

    ``normalized_class_map`` must come from :func:`build_normalized_class_map`.
    Returns ``None`` for an unmatched name (caller should count these so
    coverage is auditable, never silently drop).
    """
    return normalized_class_map.get(normalize_class_name(name))


__all__ = [
    "Box",
    "build_normalized_class_map",
    "crop_to_box",
    "match_class_to_index",
    "normalize_class_name",
    "parse_voc_xml",
]
