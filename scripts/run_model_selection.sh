#!/usr/bin/env bash
#
# LOO model selection on the real HZO data: compares Matern nu in {0.5,1.5,2.5}
# and fixed-vs-learned noise, all fit via marginal likelihood with weakly
# informative ARD lengthscale priors. Prints a ranked LOO-RMSE / LOO-NLPD table
# and saves predictions/model_selection/model_selection.png.
#
# Usage:  scripts/run_model_selection.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
python src/run_model_selection.py "$@"
