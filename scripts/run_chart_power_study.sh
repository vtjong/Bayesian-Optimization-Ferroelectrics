#!/usr/bin/env bash
# Chart-comparison power study across readout types (binary vs continuous XRD/Raman/optical).
# Saves figures + results.json to predictions/chart_power_study/
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_chart_power_study.py "$@"
