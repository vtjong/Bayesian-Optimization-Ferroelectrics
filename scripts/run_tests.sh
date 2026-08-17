#!/usr/bin/env bash
# Run the test suite (pip install -r requirements-dev.txt first).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/ "$@"
