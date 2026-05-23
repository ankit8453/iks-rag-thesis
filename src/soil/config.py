"""Pydantic schema for the soil module.

Per master reference §11 (Soil Module) the soil model is **visual-only**.
It must never output NPK, pH, fertility, organic matter %, or any other
chemical/quantitative property. The :class:`SoilConfig` schema enforces
that constraint at load time by validating ``disallowed_outputs`` against
:data:`_DISALLOWED_CHEMICAL_OUTPUTS`.

Post-Phase-4 reconciliation (supervisor sign-off received):

- **Heads kept:** ``soil_type``, ``moisture_appearance``, ``texture``.
- **Heads dropped:** ``surface``, ``cover`` — the soil-parameter
  coverage audit found neither carried IKS-corpus retrieval value.
- ``texture`` survives via IRSID label mapping (USDA texture-triangle
  ``sand/loamy_sand/clay/sandy_loam/loam`` → master plan §14
  ``coarse/fine/mixed``). See ``configs/data/soil_texture_label_mapping.yaml``.
- ``moisture_appearance`` is supervised by the Sirajganj 2025 dataset
  (Mendeley DOI 10.17632/skcc44yvvg.2, ``dry``/``moderate``/``wet``).

Phantom-fs class list verified against the upstream
``Phantom-fs/Soil-Classification-Dataset`` repo and the paper (Sheth et
al. 2025, *Engineering Applications of AI*). The seven Indian deposit
labels are Alluvial / Arid / Black / Laterite / Mountain / Red /
Yellow. The Phase 4 prompt's mention of "Clay" and "Peat" was
incorrect — those folders do not exist in the upstream dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator

from src.utils.config import BaseConfig
from src.utils.paths import CONFIGS_DIR

_DISALLOWED_CHEMICAL_OUTPUTS: frozenset[str] = frozenset(
    {
        "npk",
        "ph",
        "fertility",
        "organic_matter",
        "chemical_composition",
        "nitrogen",
        "phosphorus",
        "potassium",
        "ec",
        "salinity",
    }
)

# Post-reconciliation: surface and cover dropped. moisture renamed to
# moisture_appearance to match the master plan §14 terminology and the
# Sirajganj dataset's role.
SoilHead = Literal["soil_type", "moisture_appearance", "texture"]

_DEFAULT_SOIL_HEADS: list[SoilHead] = [
    "soil_type",
    "moisture_appearance",
    "texture",
]

# The seven actual Phantom-fs Indian deposit labels [ADDED — verified
# against upstream Phantom-fs/Soil-Classification-Dataset repo and the
# Sheth et al. 2025 paper, supervisor sign-off received]. "Clay" and
# "Peat" from the Phase 4 prompt's incorrect list do NOT exist in the
# upstream dataset and never did.
_VALID_SOIL_TYPES: list[str] = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
]


class SoilConfig(BaseConfig):
    """Configuration for the multi-task visual soil classifier."""

    backbone: Literal["efficientnet_b0"] = "efficientnet_b0"
    image_size: int = Field(224, ge=64)
    batch_size: int = Field(32, ge=1)
    lr: float = Field(1e-4, gt=0.0)
    epochs: int = Field(40, ge=1)
    seed: int = Field(42, ge=0)
    pretrained: bool = True

    multi_task_heads: list[SoilHead] = Field(
        default_factory=lambda: list(_DEFAULT_SOIL_HEADS),
        description=(
            "Visual heads only. soil_type (7-class Phantom-fs Indian "
            "deposits) + moisture_appearance (3-class Sirajganj 2025) + "
            "texture (3-class IRSID -> coarse/fine/mixed via the §14 "
            "mapping). surface and cover were dropped during the soil-"
            "parameter coverage audit."
        ),
    )

    disallowed_outputs: list[str] = Field(
        default_factory=lambda: [
            "npk",
            "ph",
            "fertility",
            "organic_matter",
            "chemical_composition",
        ],
        description=(
            "Hard list of attributes the model must never output. Enforced "
            "at config load to prevent scope creep into chemistry-from-image."
        ),
    )

    valid_soil_types: list[str] = Field(
        default_factory=lambda: list(_VALID_SOIL_TYPES),
        description=(
            "[ADDED — verified against upstream Phantom-fs repo, supervisor "
            "sign-off received] The seven Indian deposit-type labels the "
            "soil_type head emits: Alluvial / Arid / Black / Laterite / "
            "Mountain / Red / Yellow."
        ),
    )

    cross_region_validation: bool = Field(
        default=True,
        description=(
            "If True, evaluation uses a region-held-out split (validate on "
            "regions not seen during training). See master reference §11."
        ),
    )

    @field_validator("disallowed_outputs")
    @classmethod
    def _check_disallowed_subset(cls, value: list[str]) -> list[str]:
        """Every entry in ``disallowed_outputs`` must be a recognised disallowed key."""
        bad = [v for v in value if v.lower() not in _DISALLOWED_CHEMICAL_OUTPUTS]
        if bad:
            raise ValueError(
                f"Unknown disallowed_output entries: {bad}. "
                f"Allowed keys: {sorted(_DISALLOWED_CHEMICAL_OUTPUTS)}"
            )
        return value

    @field_validator("multi_task_heads")
    @classmethod
    def _heads_must_be_visual(cls, value: list[SoilHead]) -> list[SoilHead]:
        if not value:
            raise ValueError("multi_task_heads must contain at least one head.")
        return value


def load_texture_label_mapping(
    path: Path | None = None,
) -> dict[str, str]:
    """Load the IRSID-texture → master-plan-§14 mapping.

    Returns a dict mapping IRSID's USDA texture-triangle labels (e.g.
    ``sand``, ``loamy_sand``, ``clay``, ``sandy_loam``, ``loam``) to the
    master plan §14 coarse/fine/mixed scheme used by the soil module's
    ``texture`` head.

    Parameters
    ----------
    path : Path, optional
        Override the default location
        ``configs/data/soil_texture_label_mapping.yaml``.
    """
    if path is None:
        path = CONFIGS_DIR / "data" / "soil_texture_label_mapping.yaml"
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return dict(raw["texture_mapping"])
