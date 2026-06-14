#!/usr/bin/env bash
#
# Design-stage Bayesian power study (the go/no-go gate from the plan, REVISION 1 #1):
# can ~20-40 noisy points discriminate the candidate mechanism shapes? Prints P(correct)
# vs n and noise, saves predictions/power_study/power_curve.png. Synthetic — no XRD needed.
#
# Usage:  scripts/run_power_study.sh [--reps N] [--target P] [--threshold L]
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
python src/run_power_study.py "$@"
