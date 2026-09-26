# How to undo a corpus change

Every corpus change is reversible. Nothing below needs an API key or the internet.

## Safety points taken 2026-09-26 (before the citation-label fix)

| Artifact | Location |
|---|---|
| git tag of the whole repo | `thesis-safe-2026-09-26-pre-citationfix` |
| chunk JSONL files | `corpus/chunks_backup_2026-09-26_pre_citationfix/` |
| ChromaDB vector store | `corpus/vector_db_backup_2026-09-26/` |
| evaluation query set | `data/eval/silver_queries.backup_2026-09-26.json` |
| content fingerprint | `corpus/_fingerprint_pre_citationfix.json` |

Earlier safety point: `corpus/chunks_backup_pre_chapterfix/` (before the §6l chapter fix).

## Restore the corpus

```bash
rm -rf corpus/chunks corpus/vector_db
cp -r corpus/chunks_backup_2026-09-26_pre_citationfix corpus/chunks
cp -r corpus/vector_db_backup_2026-09-26 corpus/vector_db
cp data/eval/silver_queries.backup_2026-09-26.json data/eval/silver_queries.json
```

## Restore the code

```bash
git stash            # keep anything uncommitted
git checkout thesis-safe-2026-09-26-pre-citationfix
```

## Verify a rebuild did not lose content

The citation fix changes chunk **labels only** — every chunk's text must stay identical.
`scripts/verify_corpus.py` recomputes the fingerprint and compares:

```bash
python scripts/verify_corpus.py
```

PASS = the set of 233 texts is byte-identical to the fingerprint, so no content was
lost or altered. FAIL = restore from the backup above and investigate before rebuilding.
