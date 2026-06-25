"""Crop–soil suitability reference (Phase 10 interface additions).

Loads ``data/crop_soil_suitability.csv`` (166 crops, mapped to the soil model's
own vocabulary — soil 7-class, texture 3-class, moisture 3-class) and exposes:

* :func:`find` — fuzzy-match a UI/predicted crop name to a table row.
* :func:`baseline` — the Quick-mode (leaf-only) auto-fill soil reading for a
  crop: primary soil type, driest acceptable moisture, primary texture
  (the "baseline-minimum" rule specified in the source workbook).
* :func:`check_suitability` — compare a *detected* soil reading against the
  crop's suitable classes and return a conservative, honest verdict.

All values are lower-case and within the model's class sets. The mapping is an
agronomic interpretation pending expert (Dr. Sunita Pandey / ICAR-TNAU-NHB)
validation — callers should label suitability output as indicative.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CSV = Path(__file__).resolve().parent.parent / "data" / "crop_soil_suitability.csv"

# moisture ordered driest -> wettest (for the "driest acceptable" baseline rule)
_MOISTURE_ORDER = {"dry": 0, "moderate": 1, "wet": 2}

# UI / disease-label crop names -> the canonical word used in the table.
_ALIASES = {
    "corn": "maize", "paddy": "rice", "capsicum": "bell pepper",
    "bell pepper": "capsicum", "brinjal": "eggplant", "eggplant": "brinjal",
    "soybean": "soyabean", "soyabean": "soybean", "groundnut": "peanut",
    "lady finger": "okra", "bhindi": "okra",
}


@dataclass(frozen=True)
class CropSoil:
    crop: str
    category: str
    soil_types: tuple[str, ...]
    textures: tuple[str, ...]
    moistures: tuple[str, ...]
    ph: str
    temperature: str
    season: str

    @property
    def primary_soil(self) -> str:
        return self.soil_types[0] if self.soil_types else ""

    @property
    def primary_texture(self) -> str:
        return self.textures[0] if self.textures else "mixed"

    @property
    def baseline_moisture(self) -> str:
        """Driest acceptable moisture — the conservative Quick-mode default."""
        if not self.moistures:
            return "moderate"
        return min(self.moistures, key=lambda m: _MOISTURE_ORDER.get(m, 1))


def _norm(name: str) -> str:
    """Lower-case, drop parentheticals and punctuation, collapse spaces."""
    s = re.sub(r"\(.*?\)", " ", str(name or "").lower())
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _keys(raw_name: str) -> set[str]:
    """Match keys for a table row: full normalised name + parenthetical names +
    individual significant words (so 'maize' hits 'Maize (Corn)')."""
    low = str(raw_name or "").lower()
    keys: set[str] = set()
    full = _norm(low)
    if full:
        keys.add(full)
    for inside in re.findall(r"\((.*?)\)", low):
        k = _norm(inside)
        if k:
            keys.add(k)
    for w in full.split():
        if len(w) >= 3 and w not in {"and", "the", "var", "leaf"}:
            keys.add(w)
    return keys


@lru_cache(maxsize=1)
def load_table() -> tuple[CropSoil, ...]:
    if not _CSV.exists():
        return ()
    out: list[CropSoil] = []
    with _CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(CropSoil(
                crop=row["crop"],
                category=row.get("category") or "",
                soil_types=tuple(s.strip() for s in row["soil_types"].split(";") if s.strip()),
                textures=tuple(s.strip() for s in row["textures"].split(";") if s.strip()),
                moistures=tuple(s.strip() for s in row["moistures"].split(";") if s.strip()),
                ph=row.get("ph") or "", temperature=row.get("temperature") or "",
                season=row.get("season") or "",
            ))
    return tuple(out)


@lru_cache(maxsize=1)
def _index() -> dict[str, CropSoil]:
    idx: dict[str, CropSoil] = {}
    for cs in load_table():
        for k in _keys(cs.crop):
            idx.setdefault(k, cs)   # first occurrence wins (rows are ordered)
    return idx


def find(crop: str) -> CropSoil | None:
    """Best-effort match of a UI/predicted crop name to a table row."""
    q = _norm(crop)
    if not q:
        return None
    idx = _index()
    # try the whole phrase, then alias, then each word (+ its alias)
    candidates = [q, _ALIASES.get(q, q)]
    for w in q.split():
        candidates.extend([w, _ALIASES.get(w, w)])
    for c in candidates:
        if c in idx:
            return idx[c]
    return None


def _clean_soil(detected: str) -> str:
    """Normalise a model soil label like 'Alluvial_Soil' -> 'alluvial'."""
    return _norm(str(detected).replace("_", " ")).replace(" soil", "").strip()


def baseline(cs: CropSoil) -> dict[str, str]:
    """Quick-mode auto-fill soil reading for a crop (typical, not measured)."""
    return {
        "soil_type": cs.primary_soil,
        "moisture": cs.baseline_moisture,
        "texture": cs.primary_texture,
    }


def check_suitability(
    cs: CropSoil, *, soil_type: str, texture: str, moisture: str
) -> dict[str, object]:
    """Conservative crop↔detected-soil comparison. Returns per-axis booleans,
    human-readable mismatch messages, and an overall ``ok`` flag."""
    ds, dt, dm = _clean_soil(soil_type), _norm(texture), _norm(moisture)
    soil_ok = (not cs.soil_types) or (ds in cs.soil_types)
    tex_ok = (not cs.textures) or (dt in cs.textures)
    moist_ok = (not cs.moistures) or (dm in cs.moistures)
    msgs: list[str] = []
    if not soil_ok:
        msgs.append(
            f"Detected soil **{ds or '—'}** is not among the soils this crop "
            f"usually prefers ({', '.join(cs.soil_types)})."
        )
    if not moist_ok:
        msgs.append(
            f"Detected moisture **{dm or '—'}** differs from this crop's usual "
            f"need ({', '.join(cs.moistures)})."
        )
    if not tex_ok:
        msgs.append(
            f"Detected texture **{dt or '—'}** differs from this crop's usual "
            f"texture ({', '.join(cs.textures)})."
        )
    return {
        "ok": soil_ok and tex_ok and moist_ok,
        "soil_ok": soil_ok, "texture_ok": tex_ok, "moisture_ok": moist_ok,
        "messages": msgs,
    }


__all__ = ["CropSoil", "load_table", "find", "baseline", "check_suitability"]
