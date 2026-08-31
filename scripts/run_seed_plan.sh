#!/usr/bin/env bash
# Generate the campaign seed plan and the blank sheet for the laboratory.
# Neither file is regenerated once the batch is fired.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_seed_plan.py "$@"
