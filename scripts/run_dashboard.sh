#!/usr/bin/env bash
#
# Launch the local HZO dashboard.
#   scripts/run_dashboard.sh          # run locally with streamlit
#   scripts/run_dashboard.sh docker   # build + run the Docker image
#
# The Crystal Structures tab needs MP_API_KEY (free: https://materialsproject.org).
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root
MODE="${1:-local}"

if [[ "$MODE" == "docker" ]]; then
  docker build -t hzo-dashboard .
  docker run --rm -p 8501:8501 -e MP_API_KEY="${MP_API_KEY:-}" hzo-dashboard
else
  streamlit run src/dashboard/app.py
fi
