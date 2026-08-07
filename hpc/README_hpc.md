# HPC layer: SLURM job-array sweep and GPU workflow

This folder adds a high-performance computing layer on top of the controller
sweep in `src/frontier.py`. The original script trains five controllers
sequentially at a single seed. Here the same work is split into 25
independent tasks (5 dose penalties × 5 seeds) that can run in parallel as a
SLURM job array, then averaged over seeds to give a frontier with error bars.

## What each file does

| File | Purpose |
| ---- | ------- |
| `sweep_config.json` | The task grid. Numbers every (seed, w_dose) pair from 0 to 24. |
| `run_one.py` | Trains one PPO controller for one task index, evaluates it on the test split, writes `results/task_N.json`. Reuses the environment and hyperparameters from `src/frontier.py`. |
| `sbatch_sweep.sh` | SLURM array script. Launches all 25 tasks. |
| `aggregate.py` | Reads every result file, averages over seeds per dose penalty, rebuilds the frontier table and plot with error bars. |
| `gpu_run_log.md` | Evidence from a GPU run of the surrogate training (Colab T4). |

## The three commands

**1. Submit** (on a SLURM cluster)

```bash
sbatch hpc/sbatch_sweep.sh
```

Queues the 25-task array; each task trains one controller.

**2. Monitor**

```bash
squeue -u $USER
```

Per-task logs stream to `hpc/logs/task_<id>.out` and `.err`.

**3. Aggregate**

```bash
python hpc/aggregate.py
```

Rebuilds the frontier from all 25 task files and writes
`reports/frontier_aggregated.json` and `.png`.

## Running without a cluster

The pipeline runs on a single machine by emulating the array with a loop. Each
iteration sets `SLURM_ARRAY_TASK_ID` and calls the same script the cluster would:

```bash
for i in $(seq 0 24); do
    SLURM_ARRAY_TASK_ID=$i python hpc/run_one.py --task $i
done
python hpc/aggregate.py
```

This proves the full submit-to-aggregate pipeline end to end. On a real cluster
the loop is replaced by a single `sbatch` submission and the 25 tasks run in
parallel. The committed `results/` and `reports/` were produced this way on a
single CPU node.

## What the sweep shows

The aggregated frontier (`reports/frontier_aggregated.png`) traces control rate
against mean drug dose across the five dose penalties, averaged over 5 seeds.
Across the range, control rate stays roughly flat (~26-31%) while mean dose
falls sharply (~81 → ~31), so the finding is dose economy at matched control
rather than a control-rate gain. The seed-averaged error bars are what let that
"matched" claim be made honestly; the highest penalty (w_dose=0.60) is the
noisiest operating point. The seed-0 tasks reproduce `src/frontier.py` exactly,
confirming the refactor is faithful.

## GPU workflow

The training code is device-agnostic: `src/train_surrogate.py` auto-selects
CUDA when a GPU is present and runs unchanged on CPU otherwise, so the same
script runs on a laptop or a GPU node. This was demonstrated on both a local
CPU and a Colab T4; see `gpu_run_log.md`. Note that for the small surrogate MLP
the CPU is actually faster than the T4: the model is too small to amortise GPU
overhead. The GPU path is the right default for scaling up (a larger surrogate,
the Neural-ODE variant, or a cluster GPU node), not a speedup on this model.

## Status

The workflow is demonstrated on a single-node setup: the job-array logic, the
per-task training script, and the aggregation step all run end to end, and the
committed results and frontier were produced this way. The `sbatch` script is
ready to submit on a NAISS system (Berzelius, Alvis) once the cluster-specific
module lines are set. Production cluster operation is not claimed.
