# HPC layer: SLURM job-array sweep and GPU workflow

This folder adds a high-performance computing layer on top of the
controller sweep in `src/frontier.py`. The original script trains five
controllers sequentially. Here the same work is split into 25 independent
tasks (5 dose penalties x 5 seeds) that can run in parallel as a SLURM
job array, with automatic GPU selection for training.

## What each file does

| File | Purpose |
| --- | --- |
| `sweep_config.json` | The task grid. Numbers every (seed, w_dose) pair from 0 to 24. |
| `run_one.py` | Trains one PPO controller for one task index, evaluates it on the test split, writes `results/task_N.json`. Reuses the environment and hyperparameters from `src/frontier.py`. Selects GPU automatically when available. |
| `sbatch_sweep.sh` | SLURM array script. Launches all 25 tasks, one GPU each. |
| `aggregate.py` | Reads every result file, averages over seeds per dose penalty, rebuilds the frontier table and plot with error bars. |

## The three commands

### 1. Submit (on a SLURM cluster)

```bash
sbatch hpc/sbatch_sweep.sh
```

This queues the 25-task array. Each task trains one controller.

### 2. Monitor

```bash
squeue -u $USER
```

Shows which array tasks are pending, running, or complete. Per-task logs
stream to `hpc/logs/task_<id>.out` and `.err`.

### 3. Aggregate

```bash
python hpc/aggregate.py
```

Once the results are in, this rebuilds the frontier from all 25 task
files and writes `reports/frontier_aggregated.json` and
`reports/frontier_aggregated.png`.

## Running without a cluster

The pipeline runs on a single machine by emulating the array with a loop.
Each iteration sets `SLURM_ARRAY_TASK_ID` and calls the same script the
cluster would call:

```bash
for i in $(seq 0 24); do
    SLURM_ARRAY_TASK_ID=$i python hpc/run_one.py --task $i
done
python hpc/aggregate.py
```

This proves the full submit-to-aggregate pipeline end to end. On a real
cluster the loop is replaced by a single `sbatch` submission and the 25
tasks run in parallel.

## GPU workflow

`run_one.py` selects CUDA automatically when a GPU is present and falls
back to CPU otherwise, so the same script runs unchanged on a laptop or a
GPU node. The device used is recorded in each result file. GPU timing
evidence is captured in `gpu_run_log.md`.

## Status

The workflow is demonstrated on a single-node setup: the job-array logic,
the per-task training script, and the aggregation step all run end to end.
The sbatch script is ready to submit on a NAISS system (Berzelius, Alvis)
once the cluster-specific module lines are set. Production cluster
operation is not claimed.
