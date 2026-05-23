"""Parse-only smoke test for notebooks/phase5_disease_training.ipynb.

Real execution requires a GPU (Stages 1–3 train EfficientNet-B4) so
we only verify the notebook is valid JSON, all cells parse cleanly,
and the three training stages are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_phase5_notebook_exists_and_parses() -> None:
    pytest.importorskip("nbformat")
    import nbformat

    nb_path = (
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "phase5_disease_training.ipynb"
    )
    if not nb_path.is_file():
        pytest.skip(
            f"phase5_disease_training.ipynb missing — run "
            f"scripts/build_phase5_notebook.py first."
        )
    nb = nbformat.read(nb_path, as_version=4)

    # nbformat performs schema validation on read. If we got here, the
    # notebook is well-formed JSON in the v4 spec.
    assert nb["nbformat"] == 4

    # Check the three training stages appear in code cells.
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    sources = "\n".join(c["source"] for c in code_cells)
    assert "--stage pretrain" in sources, "Stage 1 (pretrain) cell missing"
    assert "--stage finetune_paddy" in sources, "Stage 2 (Paddy) cell missing"
    assert "--stage finetune_plantdoc" in sources, "Stage 3 (PlantDoc) cell missing"

    # The --resume flag must appear so resume-from-checkpoint is the
    # default behaviour.
    assert sources.count("--resume") >= 3, "Each stage cell should use --resume"


def test_phase5_notebook_has_login_cell() -> None:
    pytest.importorskip("nbformat")
    import nbformat

    nb_path = (
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "phase5_disease_training.ipynb"
    )
    if not nb_path.is_file():
        pytest.skip("notebook missing")
    nb = nbformat.read(nb_path, as_version=4)
    sources = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "notebook_login" in sources, "HF Hub login cell missing"
