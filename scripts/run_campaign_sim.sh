#!/usr/bin/env bash
# Closed-loop campaign simulator: active learning vs baselines + batch/round schedule sweep.
# Saves predictions/campaign/campaign_sim.png. Synthetic — no real data needed.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_campaign_sim.py "$@"
