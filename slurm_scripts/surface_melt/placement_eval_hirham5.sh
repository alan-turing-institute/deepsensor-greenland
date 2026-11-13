#!/bin/bash -l
#SBATCH -A orchid
#SBATCH -J giantsp_hirham5
#SBATCH -o .logs/giantsp-%j.out
#SBATCH -e .logs/giantsp-%j.err
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=GIANT@turing.ac.uk
#SBATCH --gres=gpu:1
#SBATCH -t 23:59:00 #15:30:00
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32GB
#SBATCH --exclude=gpuhost[015,016]

# environment variables
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

# submit the code
srun uv run src/ds_greenland/scripts/placement_eval.py --config-name placement_eval
