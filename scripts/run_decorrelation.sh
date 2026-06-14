#!/usr/bin/env bash
# Decorrelation arm: preheat axis breaks single-pulse collinearity -> mechanism becomes identifiable.
# Saves predictions/decorrelation/decorrelation.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_decorrelation.py
