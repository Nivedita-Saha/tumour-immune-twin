#!/bin/bash
#SBATCH --job-name=twin-sweep
#SBATCH --array=0-24
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/task_%a.out
#SBATCH --error=hpc/logs/task_%a.err

# --- Cluster environment -------------------------------------------------
# Module names vary by cluster. On NAISS systems (e.g. Berzelius, Alvis)
# check `module avail` and adjust the lines below. This project uses conda,
# so we load Anaconda and activate the `twin` environment rather than a
# plain virtualenv.

# module load Anaconda/2024.02   # adjust to the cluster's module name
# source activate twin           # or: conda activate twin

# --- Run one array task --------------------------------------------------
# SLURM sets SLURM_ARRAY_TASK_ID to each value in the --array range, so
# each task in the array trains one controller for one (seed, w_dose) pair.

python hpc/run_one.py --task ${SLURM_ARRAY_TASK_ID}
