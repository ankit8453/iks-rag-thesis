# Backup Strategy (per master reference §43)

## 3 copies of everything important
The corpus, trained models, and paper drafts must exist in three places at all times.

## 2 different media
- Local SSD: this working repository
- External drive: weekly sync of `corpus/`, `models/`, `paper/`, `thesis/`

## 1 off-site copy
- GitHub remote (this repo): code, configs, small text artifacts
- Google Drive or institutional cloud: large binaries (model checkpoints, corpus_*.npy embeddings, scanned PDFs)

## Automation
Set up a weekly cron / Task Scheduler job that rsyncs `corpus/` and `paper/` to the external drive every Friday evening. Verify quarterly that restore actually works.
