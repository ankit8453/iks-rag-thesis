"""Tiny retrieval smoke test for the Phase 3 ChromaDB collection.

Runs three hard-coded test queries against the ``iks_corpus`` collection
and prints the top-3 chunks with their source metadata. **Not** a formal
retrieval eval — Phase 7 owns that — just a confidence check that
embedding + storage + metadata round-tripped correctly after
``build_corpus``.

Run::

    python -m src.rag.corpus.query_smoke

Why a two-subprocess design
---------------------------

The original implementation imported both ``chromadb`` and
``sentence_transformers`` (with its ``torch`` dependency) into the same
Python process. On Windows that triggers a hard process death: when
``chromadb`` imports first, it pulls in ``grpc`` / ``cygrpc`` DLLs that
conflict with the MSVC runtime ``torch`` loads later, segfaulting at
the model-init stage. When ``torch`` imports first, ``cygrpc`` fails to
initialise. Either order kills the process.

The fix is to do the two halves in **separate subprocesses**:

1. Subprocess A imports only ``sentence_transformers`` and encodes the
   three test queries into a JSON file (no ChromaDB at all).
2. Subprocess B imports only ``chromadb`` and reads the pre-computed
   embeddings to query the collection.

The two-subprocess split is transparent to the caller — running
``python -m src.rag.corpus.query_smoke`` from the repo root still
produces the same human-readable output.

One more Windows-specific wrinkle: even the encoder subprocess
**segfaults at process teardown** (Windows torch DLL cleanup crash)
**after** all useful work is done — last lines logged are ``shape
(N, 1024)`` and ``WROTE``, the JSON file is fully written, then the
process exits with ``0xC0000005`` (ACCESS_VIOLATION). The
orchestrator therefore treats "output file written" as success and
ignores ``returncode != 0`` if the file is present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

COLLECTION_NAME: str = "iks_corpus"
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
TOP_K: int = 3

TEST_QUERIES: tuple[tuple[str, str], ...] = (
    ("how to treat a diseased tree", "expect Vrikshayurveda / Brihat ch.55"),
    ("signs that predict rainfall", "expect Brihat ch.21-28"),
    ("how to find underground water", "expect Brihat ch.54"),
)


# --------------------------------------------------------------------- #
# Subprocess A — encode queries via sentence-transformers
# --------------------------------------------------------------------- #


_ENCODE_SCRIPT = r"""
import json, os, sys
from sentence_transformers import SentenceTransformer
m = SentenceTransformer({model_name!r}, device='cpu')
emb = m.encode({queries!r}, normalize_embeddings=True, convert_to_numpy=True)
with open({out_path!r}, 'w', encoding='utf-8') as fh:
    json.dump({{'queries': {queries!r}, 'embeddings': emb.tolist()}}, fh)
sys.stdout.flush()
os._exit(0)
"""


# --------------------------------------------------------------------- #
# Subprocess B — open ChromaDB and query using pre-computed embeddings
# --------------------------------------------------------------------- #


_QUERY_SCRIPT = r"""
import json, os, sys
import chromadb
with open({emb_path!r}, 'r', encoding='utf-8') as fh:
    payload = json.load(fh)
client = chromadb.PersistentClient(path={db_path!r})
col = client.get_or_create_collection(name={collection!r})
n = col.count()
print(f'__COUNT__ {{n}}', flush=True)
if n == 0:
    sys.exit(2)
out_rows = []
for q, emb in zip(payload['queries'], payload['embeddings']):
    r = col.query(query_embeddings=[emb], n_results={top_k})
    docs = r['documents'][0]
    metas = r['metadatas'][0]
    dists = r['distances'][0]
    rows = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        snippet = (doc or '').replace(chr(10), ' ')
        if len(snippet) > 200:
            snippet = snippet[:197] + '...'
        src = (meta.get('source_text', '?') + ' ch.' + str(meta.get('chapter', '?'))
               + ' v.' + str(meta.get('verse_or_section', '?')))
        rows.append({{'rank': i, 'dist': float(dist), 'src': src, 'snippet': snippet,
                     'topic_tags': str(meta.get('topic_tags', '')),
                     'translator': str(meta.get('translator', ''))}})
    out_rows.append({{'query': q, 'rows': rows}})
with open({results_path!r}, 'w', encoding='utf-8') as fh:
    json.dump(out_rows, fh, indent=2)
sys.stdout.flush()
os._exit(0)
"""


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def _format_metadata_line(row: dict) -> str:
    extras: list[str] = []
    if row.get("topic_tags"):
        extras.append(f"tags=[{row['topic_tags']}]")
    if row.get("translator"):
        extras.append(f"tr={row['translator']}")
    suffix = f"  ({' '.join(extras)})" if extras else ""
    return f"{row['src']}{suffix}"


def main() -> int:
    queries = [q for q, _ in TEST_QUERIES]
    expectations = [e for _, e in TEST_QUERIES]

    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        embeddings_json = scratch_path / "query_embeddings.json"
        results_json = scratch_path / "smoke_results.json"

        # Subprocess A: encode the queries.
        encode_code = _ENCODE_SCRIPT.format(
            model_name=EMBEDDING_MODEL_NAME,
            queries=queries,
            out_path=str(embeddings_json).replace("\\", "/"),
        )
        _LOGGER.info("Encoding %d queries with %s ...", len(queries), EMBEDDING_MODEL_NAME)
        enc = subprocess.run(
            [sys.executable, "-c", encode_code],
            capture_output=True, text=True,
            env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
        )
        # On Windows the encoder process segfaults at teardown (torch DLL
        # cleanup) AFTER writing the JSON; treat "file exists" as the real
        # success signal and only fail if the JSON is missing.
        if not embeddings_json.is_file():
            print(f"FAIL: encoder did not produce embeddings file (exit={enc.returncode}).")
            print("--- encoder stdout (tail) ---")
            print(enc.stdout[-1500:])
            print("--- encoder stderr (tail) ---")
            print(enc.stderr[-1500:])
            return 1
        if enc.returncode != 0:
            _LOGGER.info(
                "Encoder subprocess exit=%d (Windows torch-teardown segfault is benign; "
                "embeddings file was written).",
                enc.returncode,
            )

        # Subprocess B: open Chroma and query with the pre-computed embeddings.
        db_path = Path("corpus/vector_db").resolve()
        query_code = _QUERY_SCRIPT.format(
            emb_path=str(embeddings_json).replace("\\", "/"),
            db_path=str(db_path).replace("\\", "/"),
            collection=COLLECTION_NAME,
            top_k=TOP_K,
            results_path=str(results_json).replace("\\", "/"),
        )
        qry = subprocess.run(
            [sys.executable, "-c", query_code],
            capture_output=True, text=True,
            env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
        )
        if qry.returncode == 2:
            print(
                f"FAIL: collection {COLLECTION_NAME!r} is empty. "
                "Run `python -m src.rag.corpus.build_corpus` first."
            )
            return 1
        # Same teardown-segfault treatment as the encoder.
        if not results_json.is_file():
            print(f"FAIL: query subprocess did not produce results file (exit={qry.returncode}).")
            print("--- query stdout (tail) ---")
            print(qry.stdout[-1500:])
            print("--- query stderr (tail) ---")
            print(qry.stderr[-1500:])
            return 1

        count_line = next(
            (ln for ln in qry.stdout.splitlines() if ln.startswith("__COUNT__")),
            None,
        )
        n_vectors = int(count_line.split()[1]) if count_line else -1
        results = json.loads(results_json.read_text(encoding="utf-8"))

    # ---- Print results in the original human-readable format ----
    print(f"Collection {COLLECTION_NAME!r}: {n_vectors} vectors\n")
    for entry, expect in zip(results, expectations, strict=True):
        print("=" * 78)
        print(f"QUERY: {entry['query']!r}")
        print(f"  ({expect})")
        print("-" * 78)
        for row in entry["rows"]:
            print(f"  [{row['rank']}] dist={row['dist']:.4f}  {_format_metadata_line(row)}")
            print(f"        {row['snippet']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
