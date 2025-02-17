#!/bin/bash
#SBATCH --job-name=batch_experiemnt
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=5:00:00

# Run your Python script
module load lang/Anaconda3 
source activate ece496b_basics
wandb agent shu4-university-of-hawaii-system/ECE491B_HW1-ece496b_basics/22uhfmoc
