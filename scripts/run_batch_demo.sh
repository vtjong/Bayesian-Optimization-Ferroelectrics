#!/usr/bin/env bash
# Batch (qEntropy) vs sequential boundary mapping for beamtime.
# Saves predictions/batch_demo/batch_demo.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_batch_demo.py "$@"
