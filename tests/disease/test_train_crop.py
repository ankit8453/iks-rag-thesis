"""Tests for the C-PD cropped-dataset indexing (no network, no torch model).

Guards that ``index_cropped_samples`` pairs XMLs to images, expands every
ground-truth box into a labelled crop sample, and skips unmatched classes
+ image-less annotations — the logic the crop-retrain depends on.
"""

from __future__ import annotations

import textwrap

from PIL import Image

from src.disease.train_crop import index_cropped_samples


def _xml(body: str) -> str:
    return textwrap.dedent(body)


def test_index_pairs_boxes_to_labels(tmp_path) -> None:
    # one image with two boxes: one matched class, one unmatched
    (tmp_path / "img1.xml").write_text(_xml("""\
        <annotation>
          <object>
            <name>Apple Scab leaf</name>
            <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
          </object>
          <object>
            <name>Unknown Weed</name>
            <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>60</ymax></bndbox>
          </object>
        </annotation>
    """), encoding="utf-8")
    Image.new("RGB", (100, 100)).save(tmp_path / "img1.jpg")

    class_map = {"Apple Scab Leaf": 0, "Apple leaf": 1}
    samples = index_cropped_samples(tmp_path, class_map)

    # only the matched box survives
    assert len(samples) == 1
    path, box, label = samples[0]
    assert path.endswith("img1.jpg")
    assert label == 0
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (0, 0, 50, 50)


def test_index_skips_xml_without_image(tmp_path) -> None:
    (tmp_path / "orphan.xml").write_text(_xml("""\
        <annotation>
          <object>
            <name>Apple Scab leaf</name>
            <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>
          </object>
        </annotation>
    """), encoding="utf-8")
    # no matching image file written
    samples = index_cropped_samples(tmp_path, {"Apple Scab Leaf": 0})
    assert samples == []


def test_index_empty_dir(tmp_path) -> None:
    assert index_cropped_samples(tmp_path, {"x": 0}) == []
