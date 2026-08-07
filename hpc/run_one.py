"""
Run a single controller-training task from the sweep grid.

One task = one (seed, w_dose) pair, looked up by index in
hpc/sweep_config.json. Trains one PPO controller on the surrogate,
evaluates it on the test split, and writes hpc/results/task_N.json.

This is the inner loop of src/frontier.py, parameterised by a single
task index so it can be launched as one element of a SLURM job array
(or looped locally to emulate the array).

Run with:
    python hpc/run_one.py --task 0
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

from stable_baselines3 import PPO

# This file lives in hpc/, but the environment code lives in src/.
# Add src/ to the import path so "from env import ..." works, exactly
# as it does when frontier.py runs from inside src/.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from env import TumourImmuneEnv, N_FAIL, T_CONTROLLED  # noqa: E402
from frontier import build_vec_env, evaluate_agent      # noqa: E402


CONFIG_PATH = os.path.join(HERE, "sweep_config.json")
RESULTS_DIR = os.path.join(HERE, "results")
MODEL_DIR = os.path.join(REPO_ROOT, "models")

# Match frontier.py so the sweep reproduces the original numbers.
N_ENVS = 8
TOTAL_TIMESTEPS = 250_000


def load_task(task_id):
    """Look up one (seed, w_dose) entry from the sweep config."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    for entry in config["tasks"]:
        if entry["task"] == task_id:
            return entry
    raise ValueError(f"Task {task_id} not found in {CONFIG_PATH}. "
                     f"Valid range is 0 to {config['n_tasks'] - 1}.")


def train_and_score(seed, w_dose, device):
    """Train one PPO controller and score it on the test split.

    Mirrors train_one() in frontier.py: same hyperparameters, same
    evaluation, but the seed is a task parameter rather than a fixed
    global constant.
    """
    tag = f"w{w_dose:.2f}".replace(".", "") + f"_s{seed}"
    norm_path = os.path.join(MODEL_DIR, f"sweep_{tag}_vecnorm.pkl")
    model_path = os.path.join(MODEL_DIR, f"sweep_{tag}.zip")

    venv = build_vec_env("train", seed, N_ENVS, w_dose, training=True)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=seed, verbose=0, device=device,
    )

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    venv.save(norm_path)
    model.save(model_path)

    test = evaluate_agent(model, "test", norm_path, w_dose)
    venv.close()

    return test, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run one sweep task.")
    parser.add_argument("--task", type=int, required=True,
                        help="Task index (0-based) into sweep_config.json.")
    args = parser.parse_args()

    entry = load_task(args.task)
    seed = entry["seed"]
    w_dose = entry["w_dose"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Task {args.task}: seed={seed}, w_dose={w_dose}, device={device}")
    print(f"Training one PPO controller for {TOTAL_TIMESTEPS:,} timesteps...")

    test, elapsed = train_and_score(seed, w_dose, device)

    result = {
        "task": args.task,
        "seed": seed,
        "w_dose": w_dose,
        "control_rate": test["control_rate"],
        "mean_dose": test["mean_dose"],
        "timesteps": TOTAL_TIMESTEPS,
        "device": device,
        "seconds": round(elapsed, 1),
    }

    out_path = os.path.join(RESULTS_DIR, f"task_{args.task}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone in {elapsed:.1f} s on {device}.")
    print(f"  control_rate = {test['control_rate']:.1f} %")
    print(f"  mean_dose    = {test['mean_dose']:.2f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
