#!/usr/bin/env bash
# Calibrate the crystallization-boundary readout against measured HZO data:
# onset temperature, thermal anchor Tmax(V,t), and the permittivity readout.
# Saves predictions/calibration/calibration.png  (reads data/; no-op if data absent).
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_calibration.py "$@"
