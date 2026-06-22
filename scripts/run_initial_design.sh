#!/usr/bin/env bash
# Round-0 Latin-hypercube seed over the (V,t) box; shows it brackets the boundary.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_initial_design.py "$@"
