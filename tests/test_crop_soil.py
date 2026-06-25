"""Tests for the crop–soil suitability reference (app/crop_soil.py)."""

from __future__ import annotations

from app import crop_soil


def test_table_loads_and_is_in_model_vocab() -> None:
    table = crop_soil.load_table()
    assert len(table) > 150, "expected ~166 crops from the thesis workbook"
    soil_ok = {"alluvial", "arid", "black", "laterite", "mountain", "red", "yellow"}
    tex_ok = {"coarse", "fine", "mixed"}
    moist_ok = {"dry", "moderate", "wet"}
    for cs in table:
        assert set(cs.soil_types) <= soil_ok, cs.crop
        assert set(cs.textures) <= tex_ok, cs.crop
        assert set(cs.moistures) <= moist_ok, cs.crop


def test_find_matches_dropdown_and_detected_names() -> None:
    assert crop_soil.find("rice") is not None          # "Rice (Paddy)"
    assert crop_soil.find("maize") is not None          # "Maize (Corn)"
    assert crop_soil.find("corn") is crop_soil.find("maize")   # alias
    assert crop_soil.find("paddy") is crop_soil.find("rice")   # parenthetical
    assert crop_soil.find("Corn rust leaf".split()[0]) is not None
    assert crop_soil.find("zzznotacrop") is None


def test_baseline_uses_primary_soil_and_driest_moisture() -> None:
    rice = crop_soil.find("rice")
    b = crop_soil.baseline(rice)
    assert b["soil_type"] == rice.soil_types[0]         # primary = first listed
    assert b["texture"] == rice.primary_texture
    # driest acceptable among the crop's moistures
    order = {"dry": 0, "moderate": 1, "wet": 2}
    assert order[b["moisture"]] == min(order[m] for m in rice.moistures)


def test_check_suitability_flags_clear_mismatch() -> None:
    rice = crop_soil.find("rice")          # prefers wet, fine, alluvial/red/...
    # a clearly wrong soil reading for rice
    bad = crop_soil.check_suitability(
        rice, soil_type="Arid_Soil", texture="coarse", moisture="dry"
    )
    assert bad["ok"] is False
    assert bad["messages"], "should explain the mismatch"
    # a matching reading passes
    good = crop_soil.check_suitability(
        rice, soil_type="Alluvial_Soil", texture="fine", moisture="wet"
    )
    assert good["ok"] is True
    assert good["messages"] == []


def test_clean_soil_handles_model_suffix() -> None:
    rice = crop_soil.find("rice")
    # "Alluvial_Soil" must normalise to "alluvial" and be accepted
    res = crop_soil.check_suitability(
        rice, soil_type="Alluvial_Soil", texture="fine", moisture="wet"
    )
    assert res["soil_ok"] is True
