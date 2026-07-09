#!/usr/bin/env bash
# Choose the LHS-seed / active-round split for the boundary-mapping campaign.
# Saves predictions/seed_budget/seed_budget.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_seed_budget.py "$@"
