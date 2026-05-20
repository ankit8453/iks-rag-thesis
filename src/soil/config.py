"""Pydantic schema for the soil module.

Per master reference §11 (Soil Module) the soil model is **visual-only**.
It must never output NPK, pH, fertility, organic matter %, or any other
chemical/quantitative property. The :class:`SoilConfig` schema enforces
that constraint at load time by validating ``disallowed_outputs`` against
:data:`_DISALLOWED_CHEMICAL_OUTPUTS`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from src.utils.config import BaseConfig

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

SoilHead = Literal["soil_type", "texture", "surface", "moisture", "cover"]

_DEFAULT_SOIL_HEADS: list[SoilHead] = [
    "soil_type",
    "texture",
    "surface",
    "moisture",
    "cover",
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
        description="Visual heads only. Each head is a separate classifier.",
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
