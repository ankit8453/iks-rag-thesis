# IKS-Grounded Multimodal Agricultural Advisory System
M.Tech Thesis · IIITDM Jabalpur · Supervisor: Dr. Akshay Pandey

## Status
Phase 1 — Foundation learning + infrastructure (Weeks 2–4 of 40). See `progress.md` for the weekly log.

## Quick start
```
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                            # verify reproducibility utilities
jupyter notebook notebooks/00_environment_check.ipynb
```

## Repository structure (per master reference §41)
```
README.md
progress.md
requirements.txt
environment.yml
.gitignore
data/{plant_disease, soil, splits}/
corpus/{raw, cleaned, chunks, vector_db}/
src/{disease, soil, rag, integration, explain, eval, utils}/
notebooks/
scripts/
configs/
results/
models/
demo/
paper/
thesis/
```

## Plan
40-week timeline split across 14 phases. See `progress.md` for week-by-week status and the master reference document for the strategic plan.

## Key guardrails
- Soil module is visual-only (texture, surface, moisture, cover, type). No NPK / pH / fertility inference from images (per master reference §14).
- All randomness is seeded via `src.utils.seeding.set_global_seed` (per master reference §37).
- Cross-region validation required for soil accuracy claims (per master reference §11/§14).
- Every commit follows version control from day 1; weekly summaries in `progress.md` (per master reference §37/§29/§44).

## Backup
See `BACKUP.md` (per master reference §43).

## Citations and bibliography
- `literature_tracker.csv` — master spreadsheet (per §42)
- `references.bib` — BibTeX export from Zotero (per §42)
