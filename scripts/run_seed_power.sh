#!/usr/bin/env bash
# Power analysis for the seed design: can the iso-Tmax ladder identify the boundary tilt?
# Saves predictions/seed_power/seed_power.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_seed_power.py "$@"
