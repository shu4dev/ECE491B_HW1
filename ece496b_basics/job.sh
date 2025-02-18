#!/bin/bash
#SBATCH --job-name=owt-train
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=5:00:00

# Run your Python script
module load lang/Anaconda3 
source activate ece496b_basics
/home/shu4/.conda/envs/ece496b_basics/bin/python /home/shu4/ECE491B_HW1/ece496b_basics/train.py @/home/shu4/ECE491B_HW1/ece496b_basics/args.txt