"""Pydantic v2 base classes for typed YAML configs.

See ADR-0002 for why we use Pydantic rather than Hydra. The short version:
every config is a frozen, strict-extra model so typos in YAML files fail
loudly instead of being silently dropped on the floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for all module configuration schemas.

    Notes
    -----
    - ``extra="forbid"`` ensures a typo in a YAML key raises a
      ``ValidationError`` instead of being silently ignored.
    - ``frozen=True`` makes config objects immutable after construction;
      mutating a config mid-run is almost always a bug.
    - Subclasses can override ``model_config`` to relax these defaults, but
      do so only with a justifying comment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def load_config(path: Path | str, schema: type[T]) -> T:
    """Load a YAML file and validate it against a Pydantic schema.

    Parameters
    ----------
    path : Path or str
        Filesystem path to the YAML file.
    schema : type[BaseConfig]
        The Pydantic schema class to validate against. Typically the
        module's own ``Config`` class.

    Returns
    -------
    BaseConfig
        A validated, frozen instance of ``schema``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pydantic.ValidationError
        If the YAML contents do not satisfy ``schema``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return schema.model_validate(raw)


def dump_config(config: BaseConfig, path: Path | str) -> None:
    """Serialise a validated config back to YAML.

    Mainly useful for snapshotting an experiment's effective configuration
    next to its results so the run is reproducible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


__all__ = ["BaseConfig", "dump_config", "load_config"]
