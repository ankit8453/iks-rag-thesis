"""Ingest Crop_Soil_Suitability_THESIS.xlsx -> a clean, version-controlled CSV.

The THESIS workbook is already mapped to the soil model's controlled vocabulary
(soil 7-class, texture 3-class, moisture 3-class), so this is a faithful
normalise-and-flatten, not an interpretation step:

  - soil_types : semicolon list from the model's 7 classes (first = primary)
  - textures   : semicolon list from {coarse, fine, mixed}
  - moistures  : semicolon list from {dry, moderate, wet}  (Excel "Moist" -> moderate)

pH / temperature / season are kept verbatim as reference context. The trailing
legend row in the sheet is skipped.
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Crop_Soil_Suitability_THESIS.xlsx"
OUT = ROOT / "data" / "crop_soil_suitability.csv"

SOIL_OK = {"alluvial", "arid", "black", "laterite", "mountain", "red", "yellow"}
TEX_OK = {"coarse", "fine", "mixed"}
MOIST_MAP = {"dry": "dry", "moist": "moderate", "moderate": "moderate", "wet": "wet"}


def _split(cell: str, mapper=None, allowed=None) -> list[str]:
    out: list[str] = []
    for part in str(cell or "").replace("/", ",").split(","):
        p = part.strip().lower()
        if not p:
            continue
        if mapper:
            p = mapper.get(p, p)
        if allowed is not None and p not in allowed:
            continue
        if p not in out:
            out.append(p)
    return out


def main() -> int:
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    data = [r for r in rows[1:] if r and r[0]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "category", "soil_types", "textures", "moistures",
                    "ph", "temperature", "season"])
        for r in data:
            soils = _split(r[2], allowed=SOIL_OK)
            texs = _split(r[3], allowed=TEX_OK)
            moists = _split(r[4], mapper=MOIST_MAP, allowed=set(MOIST_MAP.values()))
            if not soils:          # skips the trailing legend/notes row
                continue
            w.writerow([str(r[0]).strip(), r[1], "; ".join(soils),
                        "; ".join(texs), "; ".join(moists), r[5], r[6], r[7]])
            written += 1

    print(f"Wrote {written} crops -> {OUT.relative_to(ROOT)}")
    for r in data[:6]:
        soils = _split(r[2], allowed=SOIL_OK)
        if not soils:
            continue
        print(f"  {str(r[0])[:22]:22} | {'; '.join(soils):28} | "
              f"{'; '.join(_split(r[3], allowed=TEX_OK)):12} | "
              f"{'; '.join(_split(r[4], mapper=MOIST_MAP, allowed=set(MOIST_MAP.values())))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
