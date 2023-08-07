#!/bin/bash

# Replace 'notebook1.ipynb' and 'notebook2.ipynb' with the paths to your Jupyter notebooks
NOTEBOOK1="/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/src/ferro-gp_mle_bo_training.ipynb"
NOTEBOOK2="/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/src/ferro_gp_voltage_calibrations.ipynb"

# Execute the first notebook
echo "Training..."
jupyter nbconvert --to notebook --execute --inplace "$NOTEBOOK1"

# Execute the second notebook
echo "Calibrating voltages..."
jupyter nbconvert --to notebook --execute --inplace "$NOTEBOOK2"

echo "Execution complete!"
