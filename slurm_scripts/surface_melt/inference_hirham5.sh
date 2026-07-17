#!/bin/bash -l
#SBATCH -A orchid
#SBATCH -J giantsp_hirham5
#SBATCH -o .logs/giantsp-%j.out
#SBATCH -e .logs/giantsp-%j.err
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=acoca@turing.ac.uk
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00 #15:30:00
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128GB

# environment variables
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

# submit the code
srun uv run src/giant_sp/scripts/inference.py --run_ref acoca/giant-sp/4flbw8xy --config_file src/giant_sp/configs/hirham5zarr.yaml --val_set station --every 1 --weights_filename best-weights.pt --plot