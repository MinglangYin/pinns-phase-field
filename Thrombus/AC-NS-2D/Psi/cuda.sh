#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 01:00:00
#SBATCH -o sum.out

# Compile CUDA program and run
~/Environments/PyTorch-GPU/bin/python ac_ns_psi.py
