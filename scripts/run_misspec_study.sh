#!/usr/bin/env bash
# Misspecification stress test: recovery vs a wrong thermal model.
# Saves predictions/misspec_study/misspec.png + results.json
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_misspec_study.py "$@"
