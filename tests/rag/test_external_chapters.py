"""Chapter-split ingest for prepared external texts (Vishvavallabha, 6q).

Vishvavallabha restarts its verse numbering in every chapter. Chunked as one block,
verse 9 of chapter I and verse 9 of chapter VIII would share a citation.
"""

from __future__ import annotations

from src.rag.corpus.build_corpus import split_external_chapters
from src.rag.corpus.chunking import uniquify_labels

META = {"book_id": "vishvavallabha", "source_text": "Vishvavallabha", "edition": "e",
        "chapter": "pages_55_92", "original_language": "Sanskrit",
        "translator": "Nalini Sadhale", "topic_tags": [], "metadata_extras": {"x": 1}}

TEXT = """preface text that must be ignored

## Chapter 1A: Groundwater, part A

1. Water lies beneath the udumbara tree.

2. Dig three hands to the west.

## Chapter 8: Diseases and treatment

1. Trees suffer from wind, bile and phlegm.

2. Sprinkle kunapa water on a tree with wind disorder.
"""


def test_each_chapter_is_chunked_under_its_own_label() -> None:
    chunks = split_external_chapters(TEXT, META)
    assert {c.chapter for c in chunks} == {"1A", "8"}


def test_restarted_verse_numbers_do_not_collide() -> None:
    chunks = uniquify_labels(split_external_chapters(TEXT, META))
    labels = [(c.chapter, c.verse_or_section) for c in chunks]
    assert len(set(labels)) == len(labels)
    # and no letter suffix was needed - the chapter keeps them apart
    assert not any(v[-1].isalpha() for _, v in labels)


def test_text_before_the_first_marker_is_dropped() -> None:
    chunks = split_external_chapters(TEXT, META)
    assert not any("preface" in c.text for c in chunks)


def test_chapter_title_is_kept_and_other_metadata_survives() -> None:
    chunks = split_external_chapters(TEXT, META)
    c8 = [c for c in chunks if c.chapter == "8"][0]
    assert c8.metadata_extras["chapter_title"] == "Diseases and treatment"
    assert c8.metadata_extras["x"] == 1
    assert c8.translator == "Nalini Sadhale"


def test_text_without_markers_falls_back_to_one_block() -> None:
    chunks = split_external_chapters("1. One verse.\n\n2. Another verse.\n", META)
    assert chunks and all(c.chapter == "pages_55_92" for c in chunks)
