#!/usr/bin/env bash
# Earshot laptop fallback — run the entire demo with no hardware at all.
#
#   ./laptop-demo.sh
#
# Creates a venv, installs the ML package + backend, downloads the model
# (first run only, needs internet), starts the backend, and opens the
# dashboard + virtual puck in your browser. Ctrl-C stops everything.
# See docs/laptop-fallback.md for the full plan and demo script.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
PORT="${EARSHOT_PORT:-8000}"

if [ ! -d .venv-laptop ]; then
  echo "[laptop-demo] creating venv..."
  "$PY" -m venv .venv-laptop
fi
# shellcheck disable=SC1091
source .venv-laptop/bin/activate

echo "[laptop-demo] installing ML + backend deps (quiet)..."
pip install --quiet ./ml -r backend/requirements.txt

echo "[laptop-demo] fetching model artifacts (no-op if present)..."
earshot download

# Open the pages once the server is up. Reminder: macOS needs Microphone
# permission for the terminal or live detection hears silence.
(
  sleep 2
  BASE="http://localhost:${PORT}/ui"
  if command -v open >/dev/null 2>&1; then OPEN=open
  elif command -v xdg-open >/dev/null 2>&1; then OPEN=xdg-open
  else OPEN=""; fi
  if [ -n "$OPEN" ]; then
    "$OPEN" "$BASE/dashboard.html" || true
    "$OPEN" "$BASE/virtual-puck.html" || true
  else
    echo "[laptop-demo] open these in a browser:"
    echo "  $BASE/dashboard.html"
    echo "  $BASE/virtual-puck.html"
  fi
) &

echo "[laptop-demo] starting backend on port ${PORT} (Ctrl-C to stop)"
cd backend && exec python -m app.main
