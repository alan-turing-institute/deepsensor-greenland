#!/bin/bash -l
#SBATCH -A orchid
#SBATCH -J giantsp_transferlearning
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
#SBATCH --mem=256GB
#SBATCH --exclude=gpuhost015,gpuhost016

# environment variables
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

# submit the code
srun uv run src/ds_greenland/scripts/transfer.py --config-name transfer.local
