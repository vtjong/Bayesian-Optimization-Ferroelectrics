#!/usr/bin/env bash
# Extract electrical features for one round, e.g.  ./scripts/run_electrical_features.sh 2
# Re-running a round replaces its tables and its rows in the campaign table.
# Raw exports are read from data/electrical/raw/round<N>; tables go to data/electrical/features/round<N>,
# the QC report to predictions/electrical/round<N>/qc_report.html, and the rows are merged into
# data/electrical/features/campaign_per_capacitor.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
n="${1:?round number}"; shift || true
python src/run_electrical_features.py --data "data/electrical/raw/round$n" --round "$n" \
    --out "data/electrical/features/round$n" \
    --campaign data/electrical/features/campaign_per_capacitor.csv \
    --report "predictions/electrical/round$n/qc_report.html" --overwrite "$@"
