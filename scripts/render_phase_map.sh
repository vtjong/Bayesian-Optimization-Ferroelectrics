#!/usr/bin/env bash
#
# Train a GP on the experimental data and render the (time, energy) phase map +
# uncertainty map into predictions/phase_map/. Uses the electrical FOM as a stand-in
# target until XRD crystallinity labels are available.
#
# Requirements: the core deps (pip install -r requirements.txt). No MP key needed.
#
# Usage:  scripts/render_phase_map.sh [--out DIR] [--epochs N] [--threshold V]
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

python src/run_phase_map.py "$@"
