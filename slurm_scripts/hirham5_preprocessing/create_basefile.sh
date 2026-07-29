#!/bin/bash -l
#SBATCH -A aria_giant
#SBATCH -J giant_poc-cpu
#SBATCH -o .logs/giant-cpu-%j.out
#SBATCH -e .logs/giant-cpu-%j.err
#SBATCH -t 23:59:00 #15:30:00
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --mem=256GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=94

# submit the code
srun uv run src/giant_sp/scripts/hirham5_create_basefile.py 
