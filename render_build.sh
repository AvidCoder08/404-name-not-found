#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Build started..."

# Upgrade pip
python -m pip install --upgrade pip

# Install specific Python dependencies for GNN
# Note: We install torch first purely from CPU wheels to save space and ensure it's present
# before torch-scatter tries to build.
echo "Installing PyTorch (CPU)..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
# We use --no-build-isolation for GNN libs so they can see the installed torch
echo "Installing project dependencies..."
pip install -r requirements.txt
