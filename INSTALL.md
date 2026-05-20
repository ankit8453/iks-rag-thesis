# Installation

This project targets **Python 3.11–3.12** and uses `pyproject.toml` as the
single source of truth for dependencies. `requirements.txt` and
`requirements-dev.txt` are regenerated from it.

## Quick start

```bash
git clone https://github.com/ankit8453/iks-rag-thesis.git
cd iks-rag-thesis
```

### Option A — `uv` (recommended)

[`uv`](https://github.com/astral-sh/uv) is fast and pins exact resolutions.

```bash
uv venv --python 3.11
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

uv pip install -e ".[dev]"
```

### Option B — stdlib `venv` + `pip`

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"
```

### Enable git hooks

```bash
pre-commit install
```

This runs `ruff`, `black`, `mypy`, and a few file-hygiene hooks on every
commit. To run them once over the whole tree:

```bash
pre-commit run --all-files
```

## Optional extras

| Extra        | When to install                                    | Command                        |
|--------------|----------------------------------------------------|--------------------------------|
| `gpu`        | You have a CUDA GPU and want 4-bit LLM quantization | `pip install -e ".[gpu]"`      |
| `notebooks`  | You want Jupyter + TensorBoard locally             | `pip install -e ".[notebooks]"`|
| `demo`       | You want to run the Streamlit / Gradio UI          | `pip install -e ".[demo]"`     |

> `bitsandbytes` is Linux-only. On Windows, install `[gpu]` inside WSL or
> Colab; CPU-only development on Windows works without it.

## Verifying the install

```bash
python -c "from src.utils import set_global_seed, get_logger, PROJECT_ROOT; \
           set_global_seed(42); get_logger('check').info('ok'); print(PROJECT_ROOT)"
pytest -q
ruff check .
black --check .
mypy src/
```

All five commands should exit clean. If `pytest` reports failures unrelated
to `tests/utils/`, file an issue — those are the only tests that should be
green on first install (the rest are stubs and may be marked `skip`).

## Datasets and model weights

Datasets and trained weights live **outside** the repo. See
[`README.md`](README.md#datasets) for download locations and the
target paths (`data/plant_disease/`, `data/soil/`, `corpus/raw/`,
`models/`). These directories are git-ignored.
