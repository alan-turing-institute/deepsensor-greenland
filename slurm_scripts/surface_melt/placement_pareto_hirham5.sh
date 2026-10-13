#!/bin/bash -l
#SBATCH -A aria_giant
#SBATCH -J giant_poc-cpu
#SBATCH -o .logs/giant-cpu-%j.out
#SBATCH -e .logs/giant-cpu-%j.err
#SBATCH -t 01:00:00
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# submit the code
srun uv run src/giant_sp/scripts/placement_pareto.py --config-name placement_pareto.local
