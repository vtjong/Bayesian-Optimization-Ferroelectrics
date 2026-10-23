#!/usr/bin/env bash
# Write the initial design plan and the blank sheet the laboratory fills in.
# Neither file is regenerated once the batch is fired.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_initial_design.py "$@"
