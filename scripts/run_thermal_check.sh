#!/usr/bin/env bash
# Validate the thermal forward model before it feeds the campaign.
# Saves predictions/thermal_check/thermal_check.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_thermal_check.py "$@"
