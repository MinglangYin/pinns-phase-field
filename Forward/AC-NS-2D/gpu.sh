#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 3:00:00
#SBTACH -n 1
#SBATCH -o log.out
~/Environments/PyTorch-GPU/bin/python ./ac_ns_2d_dim.py
