#!/usr/bin/env bash
#
# Render HfO2 crystal-structure PNGs (monoclinic / tetragonal / polar orthorhombic /
# cubic) + a 2x2 comparison grid into predictions/structures/.
#
# Requirements:
#   - MP_API_KEY env var (free key: https://materialsproject.org)
#   - viz extras:  pip install -r requirements-viz.txt
#
# Usage:  scripts/render_structures.sh [--out DIR] [--phase KEY]
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "[error] MP_API_KEY is not set."
  echo "        Get a free key at https://materialsproject.org and run:"
  echo "        export MP_API_KEY=your_key_here"
  exit 1
fi

python src/run_structure_visualization.py "$@"
