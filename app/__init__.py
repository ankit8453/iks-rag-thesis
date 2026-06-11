"""Phase 10 Streamlit application — full-system live demo.

Wires Phases 5 (disease) + 6 (soil) + 7 (RAG) + 8 (integration) +
9 (explainability) into a single Colab-hosted UI exposed via localtunnel.

This package is import-safe — none of the submodules touch GPU / network
at import time (loaders defer that to first call). Tests under
``tests/app/`` rely on this.
"""
