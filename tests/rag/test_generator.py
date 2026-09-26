

# --------------------------------------------------------------------------- #
# Citation format (§6p). The first end-to-end run of the deployed system measured a
# valid-citation rate of 20.74%, which looked like a grounding failure and was not.
# The context block headed each passage "[Source 2] Vrikshayurveda, ch.full, v.1.2d"
# while the citation rule asked for "[Source Text, ch.X, v.Y]". "Source" appeared in
# both, so the model merged them and cited "[Source 2, ch.full, v.1.2d]" - chapter and
# verse correct, the book's name replaced by its position in the list. Such citations
# resolve to nothing even though the right passage was used.
# --------------------------------------------------------------------------- #


def _ctx_chunks():
    from types import SimpleNamespace
    return [
        SimpleNamespace(
            metadata={"source_text": "Vrikshayurveda", "chapter": "full",
                      "verse_or_section": "1.2d", "translator": "Nalini Sadhale"},
            text="Disorder, cause, symptom and remedy ...",
        ),
        SimpleNamespace(
            metadata={"source_text": "Brihat Samhita", "chapter": "28",
                      "verse_or_section": "section_2", "translator": ""},
            text="Other indications of rain are the following ...",
        ),
    ]


def test_context_block_never_labels_passages_by_position() -> None:
    """"Source 2" must not appear anywhere the model could mistake it for a name."""
    from src.rag.generator import _format_context_block
    block = _format_context_block(_ctx_chunks())
    for i in range(1, 6):
        assert f"Source {i}" not in block


def test_each_passage_header_is_itself_a_valid_citation() -> None:
    """The header is what the model is told to copy, so it must parse as a citation --
    otherwise a perfectly obedient model still produces an unresolvable one."""
    from src.rag.generator import _CITATION_RE, _format_context_block
    block = _format_context_block(_ctx_chunks())
    parsed = [(m.group("src"), m.group("chap"), m.group("verse"))
              for m in _CITATION_RE.finditer(block)]
    assert parsed == [("Vrikshayurveda", "full", "1.2d"),
                      ("Brihat Samhita", "28", "section_2")]


def test_translator_credit_does_not_break_the_citation() -> None:
    """The translator must sit OUTSIDE the brackets; inside, it would become part of
    the verse field and stop the citation resolving."""
    from src.rag.generator import _CITATION_RE, _format_context_block
    block = _format_context_block(_ctx_chunks())
    m = _CITATION_RE.search(block)
    assert m is not None and "Sadhale" not in m.group("verse")


def test_the_citation_rule_tells_the_model_to_copy_the_header() -> None:
    from src.rag.generator import SYSTEM_PROMPT_V17 as p
    assert "COPY that bracketed label exactly" in p
    assert '"Source 2" is not a citation' in p
