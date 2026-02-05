#!/bin/bash

# ----------------------------
# Create fresh venv (optional)
# ----------------------------
rm -rf ~/venvs/al4pde_scaled
python -m venv ~/venvs/al4pde_scaled/
source ~/venvs/al4pde_scaled/bin/activate
echo "Using venv at: $(which python)"
echo "Python version: $(python --version)"

# ----------------------------
# Upgrade pip + core tools
# ----------------------------
pip install --upgrade pip setuptools wheel

# ----------------------------
# Install PyTorch with CUDA 12.1 support
# ----------------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ----------------------------
# Install JAX wheel compatible with CUDA 12 + cuDNN 8.9
# Make sure to do this on a login node with internet!
# ----------------------------
pip install jax==0.4.29 jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ----------------------------
# Core scientific & logging libraries
# ----------------------------
pip install matplotlib seaborn submitit hydra-core wandb h5py pyvista bmdal_reg lightning
pip install --upgrade hydra-submitit-launcher

# ----------------------------
# Tensordict (seaborn already installed)
# ----------------------------
pip install tensordict

# ----------------------------
# Install jax-cfd
# ----------------------------
git clone https://github.com/google/jax-cfd.git
cd jax-cfd
git checkout -b new_branch d215f13
pip install -e ".[complete]"
cd ..

# ----------------------------
# Install pdearena and extras
# ----------------------------
git clone https://github.com/microsoft/pdearena
cd pdearena
pip install -e .
pip install -e ".[datagen]"
cd ..

# ----------------------------
# Install cliffordlayers from GitHub
# ----------------------------
pip install "cliffordlayers @ git+https://github.com/microsoft/cliffordlayers"
pip install "numpy<2"

# ----------------------------
# Install current package in editable mode
# ----------------------------
pip install -e .
