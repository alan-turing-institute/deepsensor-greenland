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

# environment variables
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

# submit the code
srun uv run src/ds_greenland/scripts/placement.py experiment=hirham5zarr placement.acq_fn=MeanStddev placement.n_new=10 placement.run_ref=acoca/giant-sp/4flbw8xy placement.temporal_subsampling_method=seasonal placement.temporal_subsampling_every=50 placement.search_resolution=9
