#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Build started..."

# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install PyTorch 2.4.0 (CPU) explicitly
# We use 2.4.0 as it has stable torch-scatter wheels for Python 3.11
echo "Installing PyTorch 2.4.0 (CPU)..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyG dependencies (Scatter, Sparse, Cluster) from the correct binary options
# This prevents the "Building wheel" step which fails without full dev tools
echo "Installing PyG Binaries..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

# 4. Install remaining requirements (FastAPI, Pandas, etc.)
# We exclude the GNN libs here effectively because they are satisfied, 
# but technically requirements.txt just lists names so it skips if installed.
echo "Installing project requirements..."
pip install -r requirements.txt
