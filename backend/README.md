# Shortlist Backend (FastAPI)

A thin FastAPI layer that exposes the deterministic ranking pipeline
(`ranking/` + `rank.py`) to the React/Vite frontend in `frontend/`.

It runs the pipeline **in-process** (no subprocess, no network, no LLM) and
adapts the real component scores into the "AI panel" candidate shape the UI
renders. Every derived field (panel scores, verdict, strengths/concerns,
counterfactual delta) is a deterministic, explainable function of the
candidate's actual scores.

## How it connects to the frontend

`frontend/vite.config.ts` proxies `/api` → `http://localhost:8080`, so the
frontend talks to this backend with no code changes. `frontend/src/api.ts`
defines the exact endpoints implemented here.

```
React (Vite :5173)  ──/api/*──▶  Vite proxy  ──▶  FastAPI (:8080)  ──▶  ranking pipeline
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/candidates` | All ranked candidates (runs pipeline on first call) |
| GET | `/api/candidates/{id}` | One enriched candidate |
| GET | `/api/audit` | Counterfactual fairness audit |
| GET | `/api/job-description` | The role being ranked against |
| GET | `/api/pipeline/status` | `{ running, last_result }` |
| POST | `/api/run` | Trigger a (re)run |
| POST | `/api/upload/resumes` | Stage candidate files (`.jsonl`/`.json` become the next source) |
| POST | `/api/upload/job-description` | Stage a JD file |
| GET | `/api/export/csv` | Download `submission.csv` |
| GET | `/api/health` | Liveness check |

## Run it

From the repository root:

```bash
# 1. Install backend deps (into the same venv as the ranker)
.venv/bin/pip install -r backend/requirements.txt

# 2. Start the API (port 8080)
.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080
#   or: .venv/bin/python -m backend.app

# 3. In another terminal, start the frontend (port 5173)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. The first dashboard load runs the full pipeline on
`candidates.jsonl` (~11s) and caches the result; subsequent loads are instant.
If `candidates.jsonl` is absent, the backend falls back to
`sample_candidates.json`.

## Notes

- Scores are computed in `[0, 1]` by the ranker and presented on a `0-10` scale
  in the API/UI.
- The pipeline also writes `submission.csv` on every run so the CLI, the export
  endpoint, and the UI stay in sync.
- Uploaded datasets land in `uploads/`; uploading a `.jsonl`/`.json` candidate
  file makes the next run use it as the source.
