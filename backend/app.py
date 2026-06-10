"""FastAPI application exposing the ranking pipeline to the React frontend.

Endpoints (all under ``/api``, matching ``frontend/src/api.ts``):

    GET  /api/candidates              -> { candidates: [...], total }
    GET  /api/candidates/{id}         -> single enriched candidate
    GET  /api/audit                   -> fairness/counterfactual audit report
    GET  /api/job-description         -> the role being ranked against
    GET  /api/pipeline/status         -> { running, last_result }
    POST /api/run                     -> trigger a (re)run of the pipeline
    POST /api/upload/resumes          -> accept candidate files (jsonl/json)
    POST /api/upload/job-description  -> accept a JD file
    GET  /api/export/csv              -> download submission.csv

Run with:
    uvicorn backend.app:app --host 0.0.0.0 --port 8080
or:
    python -m backend.app
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.service import BIAS_FLAG_THRESHOLD, REPO_ROOT, service

app = FastAPI(title="Shortlist Backend", version="1.0.0")

# Allow the Vite dev server (and the production preview) to call the API
# directly in case the proxy is bypassed. The Vite proxy in vite.config.ts
# routes /api -> http://localhost:8080 in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where uploaded files are staged.
UPLOAD_DIR = REPO_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
@app.get("/api/candidates")
def get_candidates() -> dict:
    """Return all ranked candidates (running the pipeline on first access)."""
    service.ensure_loaded()
    candidates = [c.to_dict() for c in service.candidates()]
    return {"candidates": candidates, "total": len(candidates)}


@app.get("/api/candidates/{candidate_id}")
def get_candidate(candidate_id: str) -> dict:
    service.ensure_loaded()
    candidate = service.candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
    return candidate.to_dict()


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------
@app.get("/api/audit")
def get_audit() -> dict:
    """Build the counterfactual fairness audit from the enriched candidates."""
    service.ensure_loaded()
    candidates = service.candidates()

    flagged = []
    failures = []
    for c in candidates:
        original = c.composite_score
        delta = c.counterfactual_delta
        cf_score = (
            round(original - (delta or 0.0) * 10, 2) if delta is not None else None
        )
        record = {
            "candidate_id": c.candidate_id,
            "name": c.name,
            "delta": delta,
            "original_score": round(original, 2),
            "cf_score": cf_score,
            "bias_flag": c.bias_flag,
            "audit_failure": False,
            "error": "",
        }
        if c.bias_flag:
            flagged.append(
                {
                    "candidate_id": c.candidate_id,
                    "name": c.name,
                    "delta": delta,
                    "original_score": round(original, 2),
                    "cf_score": cf_score,
                }
            )
        failures.append(record)

    total = len(candidates)
    flagged_count = len(flagged)
    clean_count = total - flagged_count
    flag_rate = (flagged_count / total) if total else 0.0

    return {
        "total_candidates_audited": total,
        "flagged_count": flagged_count,
        "flag_rate": round(flag_rate, 4),
        "bias_flag_threshold": BIAS_FLAG_THRESHOLD,
        "flagged_candidates": flagged,
        "clean_candidates_count": clean_count,
        "methodology_note": (
            "Counterfactual fairness check. The ranker never reads candidate "
            "names or gender, so name and pronoun swaps produce a zero score "
            "delta by construction. The institution-tier swap (setting tier to "
            "'unknown') measures real sensitivity to prestige signals; a delta "
            f"above {BIAS_FLAG_THRESHOLD} flags the candidate for manual review."
        ),
        "audit_failures": failures,
    }


# ---------------------------------------------------------------------------
# Job description
# ---------------------------------------------------------------------------
@app.get("/api/job-description")
def get_job_description() -> dict:
    """Return the role candidates are ranked against."""
    return {
        "job_id": "redrob-senior-ai-engineer",
        "title": "Senior AI Engineer — Founding Team",
        "company": "Redrob AI",
        "raw_text": (
            "Senior AI Engineer (5-9 years) for a Series A AI-native talent "
            "intelligence platform in Pune/Noida, India (hybrid). Owns the "
            "ranking, retrieval, and matching intelligence layer. Must have "
            "production embeddings-based retrieval, vector databases / hybrid "
            "search, strong Python, and rigorous ranking-evaluation experience "
            "(NDCG, MRR, MAP). Nice to have: LLM fine-tuning (LoRA/QLoRA/PEFT), "
            "learning-to-rank, HR-tech, distributed systems. Explicitly not a "
            "fit: pure researchers without production deployment, recent-only "
            "LangChain/OpenAI users, title-chasers, entirely-consulting careers, "
            "and CV/speech/robotics specialists without NLP/IR exposure."
        ),
        "requirements": [
            {"text": "Production embeddings-based retrieval systems", "bucket": "must_have", "dimension": "skill"},
            {"text": "Vector databases / hybrid search infrastructure", "bucket": "must_have", "dimension": "skill"},
            {"text": "Strong Python", "bucket": "must_have", "dimension": "skill"},
            {"text": "Ranking evaluation frameworks (NDCG, MRR, MAP)", "bucket": "must_have", "dimension": "skill"},
            {"text": "5-9 years total experience", "bucket": "must_have", "dimension": "experience"},
            {"text": "Product-company (not services) applied ML background", "bucket": "must_have", "dimension": "career"},
            {"text": "LLM fine-tuning (LoRA / QLoRA / PEFT)", "bucket": "nice_to_have", "dimension": "skill"},
            {"text": "Learning-to-rank models", "bucket": "nice_to_have", "dimension": "skill"},
            {"text": "HR-tech / recruiting / marketplace products", "bucket": "nice_to_have", "dimension": "career"},
            {"text": "Located in or willing to relocate to Pune/Noida", "bucket": "nice_to_have", "dimension": "location"},
        ],
    }


# ---------------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------------
@app.get("/api/pipeline/status")
def pipeline_status() -> dict:
    return {"running": service.running, "last_result": service.last_result}


@app.post("/api/run")
def run_pipeline() -> dict:
    if service.running:
        return {"message": "Pipeline already running"}
    result = service.run()
    if result.get("returncode") != 0:
        raise HTTPException(status_code=500, detail=result.get("error", "Pipeline failed"))
    return {"message": "Pipeline completed"}


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
@app.post("/api/upload/resumes")
async def upload_resumes(files: List[UploadFile] = File(...)) -> dict:
    """Accept candidate files. A .jsonl or .json file becomes the next source."""
    saved: List[str] = []
    for f in files:
        dest = UPLOAD_DIR / Path(f.filename).name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
        # If a candidate dataset was uploaded, point the next run at it.
        if dest.suffix.lower() in (".jsonl", ".json"):
            service._state.source_path = str(dest)  # noqa: SLF001 (intentional)
    return {"uploaded": saved}


@app.post("/api/upload/job-description")
async def upload_job_description(file: UploadFile = File(...)) -> dict:
    dest = UPLOAD_DIR / Path(file.filename).name
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"uploaded": file.filename}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.get("/api/export/csv")
def export_csv() -> FileResponse:
    service.ensure_loaded()
    path = service.output_csv_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="No submission.csv available yet")
    return FileResponse(
        path,
        media_type="text/csv",
        filename="submission.csv",
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
