#!/usr/bin/env bash
#
# Synthetic demo: a learnable input warp on (voltage, time) recovers energy-space
# smoothness (REVISION 1 #11). Fits 3 GPs and shows warped (V,t) ~ energy oracle >> raw
# (V,t), plus that the learned warp recovers the true V->energy map. Saves
# predictions/warp_demo/warp_demo.png. Synthetic — no real data needed.
#
# Usage:  scripts/run_warp_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
python src/run_warp_demo.py
