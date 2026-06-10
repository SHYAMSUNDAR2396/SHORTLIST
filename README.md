# Shortlist

An intelligent candidate ranking system that scores and ranks job applicants
against a role's requirements using structured heuristic analysis — no LLMs at
inference time, fully deterministic, and designed to run locally on a laptop.

## What it does

Given a pool of candidate profiles (as JSONL) and a job description, Shortlist:

1. **Parses and validates** candidate data (career history, skills, education, behavioral signals)
2. **Detects and excludes** fraudulent/honeypot profiles with impossible credentials
3. **Scores candidates** across six dimensions — skill relevance, career fit, experience alignment, platform engagement, education quality, and location/work-mode compatibility
4. **Applies disqualification rules** for clearly unfit candidates (all-consulting careers, keyword stuffers, domain mismatches)
5. **Ranks the top 100** with a deterministic composite score and a human-readable explanation per candidate
6. **Audits for fairness** via counterfactual sensitivity testing (institution-tier perturbation)

The system surfaces the right candidates by reading *career history and behavior*, not just keyword overlap — distinguishing someone who built a recommendation system at a product company from someone who listed "recommendation" as a skill.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (:5173)                        │
│   Dashboard · Candidate Detail · Audit Report · Pipeline Setup  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ /api (Vite proxy)
┌───────────────────────────────▼─────────────────────────────────┐
│                    FastAPI Backend (:8080)                       │
│   Enriches scores into panel view · Serves audit · CSV export   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ in-process
┌───────────────────────────────▼─────────────────────────────────┐
│                    Ranking Pipeline (Python)                     │
│                                                                 │
│  DataLoader ─► HoneypotDetector ─► 6 Scorers ─► Composite ─►   │
│  DisqualifierFilter ─► Ranker ─► ReasoningGenerator ─► CSV      │
└─────────────────────────────────────────────────────────────────┘
```

### Scoring Dimensions

| Dimension | Weight | What it measures |
|---|---|---|
| Skill Relevance | 35% | Match against must-have and nice-to-have skill groups, with trust penalties for inflated durations |
| Career Fit | 25% | Production experience at product companies, AI/ML role descriptions, title relevance, consulting/job-hopping penalties |
| Experience Fit | 15% | Alignment with the 5–9 year target range (piecewise decay outside) |
| Behavioral Signals | 10% | Recruiter response rate, GitHub activity, platform recency |
| Education | 10% | Institution tier, degree level, field relevance to AI/ML |
| Location & Work Mode | 5% | Geographic fit + hybrid/onsite preference alignment |

### Key Design Choices

- **Rule-based, not LLM-per-candidate** — transparent, auditable, and runs 100K candidates in ~11 seconds on CPU
- **Deterministic** — same input always produces byte-identical output (eval date derived from data, not wall clock)
- **Career-history inference** — reads role descriptions for production deployment signals rather than trusting skill keyword lists
- **Honeypot detection** — four temporal-consistency rules catch impossible profiles (tenure exceeding company age, skill duration exceeding career span, etc.)
- **Fairness-aware** — the scorer never reads candidate names or gender; a counterfactual audit measures sensitivity to education-tier prestige

## Quick Start

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r backend/requirements.txt

# Run the ranking pipeline (CLI)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Or start the full web app
./run_dev.sh
# Backend → http://localhost:8080
# Frontend → http://localhost:5173
```

For the frontend only:
```bash
cd frontend && npm install && npm run dev
```

## Project Structure

```
├── rank.py                  # CLI entry point
├── ranking/                 # Core pipeline
│   ├── loader.py            # Streaming JSONL parser
│   ├── honeypot.py          # Fraud detection
│   ├── scorers/             # 6 independent scoring modules
│   ├── composite.py         # Weighted score aggregation
│   ├── disqualifier.py      # JD-based penalty rules
│   ├── ranker.py            # Deterministic sort + selection
│   ├── reasoning.py         # Per-candidate explanation
│   └── formatter.py         # CSV output
├── backend/                 # FastAPI service (connects pipeline to UI)
├── frontend/                # React + Vite + Tailwind dashboard
└── tests/                   # 129 tests (property-based + unit + integration)
```

## Testing

```bash
pytest -q                    # full suite (129 tests)
pytest tests/property -v     # 22 Hypothesis property tests
pytest -m "not slow"         # skip full-dataset benchmarks
```

Property-based tests validate scoring invariants (bounded outputs, monotonic rankings, formula correctness). Integration tests verify determinism, no-network compliance, and output format validity.

## Performance

| Metric | Result |
|---|---|
| 100K candidates ranked | ~11 seconds |
| Peak memory | ~1.1 GB |
| GPU required | No |
| Network during ranking | No |
| Output reproducibility | Bit-identical |

## Tech Stack

- **Pipeline:** Python 3.12, pure standard library + dataclasses
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS, Lucide icons
- **Testing:** Pytest, Hypothesis (property-based testing)
