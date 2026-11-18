# AGENTS.md

## What this repo is

One small service: FastAPI semantic search over harvested Universitas Diponegoro (Undip) eprints. Flat layout, no packages, no tests, no CI, no lockfile. Python 3.11.

- `backend.py` — hardened version: async endpoint, embedding/query offloaded via `run_in_threadpool`, explicit `RuntimeError` at startup when the Chroma collection is missing.
- `backend_example.py` — original sync version. **The Dockerfile copies and serves this file** (`CMD uvicorn backend_example:app`), not `backend.py`. Editing `backend.py` changes nothing in the container until the Dockerfile is updated too. Keep the two in sync on the API contract and constants.
- `data_harvesting.ipynb` — the only data ingestion path. Not run by the app, not in the Docker image (`.dockerignore` excludes `*.ipynb`).

## Data prerequisites (the #1 startup failure)

- `./eprints_vector_db` is gitignored and required at startup. If absent, the app fails to boot (`RuntimeError` in `backend.py`, `chromadb.errors.NotFoundError` in `backend_example.py`).
- Ingest by running the notebook top to bottom: harvest OAI-PMH (`https://v.undip.ac.id/cgi/oai2`, `metadataPrefix=oai_dc`, target 1000 valid records) → writes `eprints_undip_clean.json` → embeds `"{title}. {description}"` and inserts into Chroma.
- The notebook **deletes and recreates** collection `undip_research_search` with `{"hnsw:space": "cosine"}`. Re-running wipes existing vectors; there is no incremental update path.
- Inserts are batched at 1000 because Chroma rejects larger batches (~5461 cap).
- Notebook needs `sickle`, which is **not** in `requirements.txt` (installed via `%pip install` cell). Notebook also calls `%pip install sentence-transformers chromadb`.
- Embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) downloads at startup. Docker sets `HF_HOME=/model-cache` backed by the named `model_cache` volume so restarts don't re-download; local runs use the default HF cache.
- Ingestion and the app must use the same model, or stored vectors won't match query vectors.

## Commands

```bash
pip install -r requirements.txt
uvicorn backend:app --reload --port 8000   # local dev; run from repo root
docker compose up --build                  # serves backend_example.py on :8000
```

- Run uvicorn from the repo root: `DB_PATH = "./eprints_vector_db"` is relative to the CWD.
- No test runner, linter, formatter, or typechecker is configured. Do not assume `pytest` / `npm test` / `ruff` exist; verify before adding them.
- `requirements.txt` is range-pinned (`chromadb>=1.5,<2`, `sentence-transformers>=6,<7`); code depends on Chroma 1.x APIs. `docker-compose.yml` bind-mounts `./eprints_vector_db`, so the host DB is what the container reads.

## API contract

- Single endpoint: `POST /search`. Request `{query: str, min_length=1, max_threshold: float = 0.5, ge=0, le=2}`.
- Response wrapper `{query, max_threshold, count, results}`; each result has **Indonesian keys**: `peringkat`, `skor`, `judul`, `penulis`, `deskripsi`, `url`, `date`.
- `skor` is a cosine **distance** (0 = identical), not a similarity, despite the name. Rows with `distance > max_threshold` are dropped and `peringkat` is a dense rank computed after filtering (so it can differ from the raw Chroma index).
- Always fetches `N_RESULTS = 100` then filters in Python. There is no pagination or `offset`.
- Result construction indexes Chroma metadata directly (`metadata["title"]`, `["creators"]`, `["url"]`, `["description"]`, `["date"]`) — a missing metadata key raises `KeyError` per request. Metadata shape is owned by the ingestion notebook; change both together.
