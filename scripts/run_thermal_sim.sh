#!/usr/bin/env bash
# Simulate flash-anneal T(t) and the thermal descriptors from (V,t).
# Saves predictions/thermal_sim/{profiles,descriptor_maps,collinearity}.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_thermal_sim.py
