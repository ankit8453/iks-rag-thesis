# ADR-0002 — Pydantic v2 over Hydra for configs

## Status

Accepted — 2026-05-20.

## Context

We need a configuration story that is:

1. **Typed** — the configs are wide (image size, learning rate,
   retrieval depth, RAGAS metric list, ...). Untyped dicts will rot.
2. **Strict** — a typo in a YAML key must fail loudly. Silently dropped
   keys are a debugging nightmare on long training runs.
3. **Debuggable** — when a config is wrong, the error message should
   point at the wrong key and line.
4. **Lightweight** — this is a one-student M.Tech project, not a
   research lab with 50 sweeps to coordinate.

Two reasonable options:

- **Hydra** (+ OmegaConf) — purpose-built for ML configs, supports
  composition and CLI overrides out of the box, used widely in
  Facebook AI Research code.
- **Pydantic v2** + plain YAML loader — schemas defined as Python
  classes, ``extra="forbid"`` + ``frozen=True`` gives us the strictness
  we want, errors are clear, no `_target_` magic.

## Decision

Pydantic v2 with frozen, ``extra="forbid"`` base class
(`src/utils/config.py::BaseConfig`). Configs are validated by an
explicit `load_config(path, schema)` helper.

## Consequences

- No Hydra composition. We accept this — at our scale we're more
  likely to copy a YAML and edit it than to compose four overlays.
- No CLI overrides via dotted keys. We accept this — scripts use
  argparse for the few flags that vary day-to-day.
- All configs are immutable. Code that wants to "tweak the LR mid-
  training" must construct a new config object. This is a feature.
- Pydantic v2's improved performance over v1 means the per-load cost
  is negligible even for the big RAG config.
- Mypy/ruff understand Pydantic models, so editor support is good.

## References

- Pydantic v2 docs.
- Hydra docs (alternative considered).
