#!/usr/bin/env bash
# Start the Shortlist backend (FastAPI :8080) and frontend (Vite :5173) together.
# Usage: ./run_dev.sh   (Ctrl-C stops both)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Expected venv python at $PYTHON. Create it with: python -m venv .venv && .venv/bin/pip install -r requirements.txt -r backend/requirements.txt"
  exit 1
fi

# Start backend.
"$PYTHON" -m uvicorn backend.app:app --host 0.0.0.0 --port 8080 &
BACKEND_PID=$!

# Start frontend.
( cd frontend && npm run dev ) &
FRONTEND_PID=$!

cleanup() {
  echo "Stopping..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Backend  -> http://localhost:8080  (PID $BACKEND_PID)"
echo "Frontend -> http://localhost:5173  (PID $FRONTEND_PID)"
echo "Press Ctrl-C to stop both."
wait
