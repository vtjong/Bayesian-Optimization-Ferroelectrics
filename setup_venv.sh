#!/bin/bash
# Setup script for virtual environment

echo "========================================="
echo "Setting up Virtual Environment"
echo "========================================="

# Create venv
echo "Creating virtual environment..."
python3 -m venv venv

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install GPyTorch dependencies first (they're needed by botorch)
echo "Installing core dependencies..."
pip install --no-cache-dir torch scipy numpy

# Install gpytorch
echo "Installing GPyTorch..."
pip install --no-cache-dir gpytorch

# Install botorch
echo "Installing BoTorch..."
pip install --no-cache-dir botorch

# Install remaining packages
echo "Installing other dependencies..."
pip install --no-cache-dir matplotlib pandas scikit-learn plotly adjustText wandb pyyaml

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the training script, run:"
echo "  cd src && python train_clean.py"
echo ""

