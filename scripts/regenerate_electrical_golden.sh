#!/usr/bin/env bash
# Rebuild the Round-2 golden tables that tests/test_electrical_round2.py compares against.
# Run only after an intended change to the feature definitions, and review the diff:
# every cell that moved is a behaviour change.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$(mktemp -d)"
python src/run_electrical_features.py --data data/electrical/raw/round2 --round 2 --out "$out" \
    --report "$out/qc_report.html"
cp "$out/features_per_capacitor.csv" tests/data/electrical/golden/round2_per_capacitor.csv
cp "$out/features_per_sample.csv" tests/data/electrical/golden/round2_per_sample.csv
rm -rf "$out"
git status --short tests/data/electrical/golden
