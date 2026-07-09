#!/usr/bin/env bash
# Functional-trace regression demo: recover an unlisted temperature-window controller.
# Saves predictions/functional_demo/functional_demo.png
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_functional_demo.py "$@"
