#!/usr/bin/env bash
# Generate the seed flash-condition plan for the boundary-mapping campaign.
# Saves data/flash_plan_seed.csv + predictions/flash_plan/flash_plan.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_flash_plan.py "$@"
