#!/usr/bin/env bash
# Acquisition comparison for noisy boundary mapping: entropy vs BALD vs level-set entropy.
# Saves predictions/picker_demo/picker_acquisition_comparison.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_picker_demo.py "$@"
