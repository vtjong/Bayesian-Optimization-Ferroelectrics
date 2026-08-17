#!/bin/bash
# Create the virtual environment for the crystallization-boundary campaign.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements-dev.txt

echo
echo "Done. Activate with:"
echo "  source venv/bin/activate"
echo "Then check the models:"
echo "  ./scripts/run_tests.sh"
