"""
Trace the control-versus-dose frontier (step A4.2, extended).

Why
---
A single trained controller gives one point on the trade-off between how many
patients are controlled and how much drug is used. On 46 test patients, one
patient moves the control rate by 2.2 points, so a single-point comparison is
suggestive rather than conclusive.

Sweeping the dose penalty W_DOSE produces a family of controllers, each
sitting at a different point on that trade-off. Plotting them against the
constant-dose baselines answers a stronger question: does learned control
dominate constant dosing across the whole operating range, or only at one
convenient setting?

A frontier that sits above and to the left of the baseline curve along its
length is a far more robust claim than any single comparison, and it is
insensitive to the exact test-set size.

Training one controller takes about a minute, so this costs a few minutes
in total.

Run with:
    python src/frontier.py
"""

import os
import json
import time
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import (
    TumourImmuneEnv, evaluate_policy,
    V_MAX, N_FAIL, T_CONTROLLED,
)


MODEL_DIR = "models"
FIGURE_DIR = "figures"
REPORT_DIR = "reports"

RANDOM_SEED = 0
N_ENVS = 8
TOTAL_TIMESTEPS = 250_000        # the earlier run plateaued well before this

# Dose penalty values to sweep. Low values buy control rate at the cost of
# drug; high values force economy.
W_DOSE_VALUES = [0.03, 0.08, 0.15, 0.30, 0.60]

torch.set_num_threads(1)


def build_vec_env(split, seed, n_envs, w_dose, norm_stats=None, training=True):
    """Build a vectorised environment with a given dose penalty."""
    def _init(rank):
        def _f():
            return TumourImmuneEnv(split=split, include_params=True,
                                   seed=seed + rank, w_dose=w_dose)
        return _f

    venv = DummyVecEnv([_init(i) for i in range(n_envs)])
    if norm_stats is None:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        venv = VecNormalize.load(norm_stats, venv)
    venv.training = training
    venv.norm_reward = False
    return venv


def evaluate_agent(model, split, norm_stats_path, w_dose):
    """Score a trained policy on every patient in a split, deterministically."""
    venv = build_vec_env(split, RANDOM_SEED + 999, 1, w_dose,
                         norm_stats=norm_stats_path, training=False)
    raw_env = venv.venv.envs[0]
    n_patients = len(raw_env.patient_ids)

    results = []
    for index in range(n_patients):
        venv.reset()
        raw_obs, _ = raw_env.reset(options={"patient_index": index})
        obs = venv.normalize_obs(raw_obs.reshape(1, -1))

        done, info = False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            raw_obs, reward, terminated, truncated, info = raw_env.step(action[0])
            obs = venv.normalize_obs(raw_obs.reshape(1, -1))
            done = terminated or truncated

        outcome = "host failure" if info["min_N"] < N_FAIL else (
            "controlled" if info["tumour"] < T_CONTROLLED else "escape"
        )
        results.append({
            "outcome": outcome,
            "total_dose": info["total_dose"],
            "final_T": info["tumour"],
        })

    venv.close()

    n = len(results)
    controlled = sum(1 for r in results if r["outcome"] == "controlled")

    return {
        "n": n,
        "controlled": controlled,
        "control_rate": 100.0 * controlled / n,
        "mean_dose": float(np.mean([r["total_dose"] for r in results])),
        "dose_std": float(np.std([r["total_dose"] for r in results])),
    }


def binomial_interval(successes, n, z=1.96):
    """
    Wilson score interval for a proportion.

    With 46 test patients a point estimate alone overstates precision, so the
    interval is reported alongside it. The Wilson form is used because the
    simple normal approximation behaves poorly at small n and extreme rates.
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100.0 * max(0.0, centre - spread), 100.0 * min(1.0, centre + spread)


def train_one(w_dose, verbose=True):
    """Train a single controller at a given dose penalty and score it."""
    tag = f"w{w_dose:.2f}".replace(".", "")
    norm_path = os.path.join(MODEL_DIR, f"frontier_{tag}_vecnorm.pkl")
    model_path = os.path.join(MODEL_DIR, f"frontier_{tag}.zip")

    venv = build_vec_env("train", RANDOM_SEED, N_ENVS, w_dose, training=True)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=RANDOM_SEED, verbose=0,
    )

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    venv.save(norm_path)
    model.save(model_path)

    val = evaluate_agent(model, "val", norm_path, w_dose)
    test = evaluate_agent(model, "test", norm_path, w_dose)
    venv.close()

    if verbose:
        print(f"  W_DOSE = {w_dose:<5.2f}  trained in {elapsed:5.1f} s   "
              f"val {val['control_rate']:5.1f} %   "
              f"test {test['control_rate']:5.1f} % at dose {test['mean_dose']:6.2f}")

    return {"w_dose": w_dose, "val": val, "test": test, "model_path": model_path}


def plot_frontier(agents, baselines):
    """The frontier plot: control rate against drug used."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Main frontier.
    ax = axes[0]

    b_dose = [b["mean_dose"] for b in baselines]
    b_rate = [b["control_rate"] for b in baselines]
    order = np.argsort(b_dose)
    ax.plot(np.array(b_dose)[order], np.array(b_rate)[order],
            "o-", color="#888888", linewidth=1.8, markersize=8,
            label="constant dose baselines", zorder=2)
    for b in baselines:
        ax.annotate(b["label"], (b["mean_dose"], b["control_rate"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=7,
                    color="#666666")

    a_dose = [a["test"]["mean_dose"] for a in agents]
    a_rate = [a["test"]["control_rate"] for a in agents]
    order = np.argsort(a_dose)
    ax.plot(np.array(a_dose)[order], np.array(a_rate)[order],
            "*-", color="#a4243b", linewidth=2.2, markersize=15,
            markeredgecolor="white", markeredgewidth=1.0,
            label="learned controllers", zorder=4)

    # Confidence intervals, since the test set is small.
    for a in agents:
        lo, hi = binomial_interval(a["test"]["controlled"], a["test"]["n"])
        ax.plot([a["test"]["mean_dose"]] * 2, [lo, hi],
                color="#a4243b", alpha=0.35, linewidth=1.4, zorder=3)
        ax.annotate(f"W={a['w_dose']:.2f}",
                    (a["test"]["mean_dose"], a["test"]["control_rate"]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=7, color="#a4243b")

    ax.set_xlabel("mean cumulative dose")
    ax.set_ylabel("control rate on unseen patients (%)")
    ax.set_title("Control against drug used\nup and to the left is better")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # Drug saving at matched control rate.
    ax = axes[1]
    labels, savings = [], []
    for a in agents:
        rate = a["test"]["control_rate"]
        # Cheapest baseline achieving at least this control rate.
        matching = [b for b in baselines if b["control_rate"] >= rate - 1e-9]
        if not matching:
            continue
        cheapest = min(matching, key=lambda b: b["mean_dose"])
        if cheapest["mean_dose"] <= 0:
            continue
        saving = 100.0 * (1 - a["test"]["mean_dose"] / cheapest["mean_dose"])
        labels.append(f"W={a['w_dose']:.2f}\n({rate:.0f} %)")
        savings.append(saving)

    if savings:
        colours = ["#1b7a5a" if s > 0 else "#a4243b" for s in savings]
        ax.bar(range(len(savings)), savings, color=colours, alpha=0.9)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("drug saved vs cheapest baseline\nat equal or better control (%)")
        ax.set_title("Drug economy at matched outcome")
        for i, s in enumerate(savings):
            ax.text(i, s, f"{s:+.1f} %", ha="center",
                    va="bottom" if s > 0 else "top", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no baseline matched at these control rates",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    fig.suptitle("Learned control dominates constant dosing across the frontier (A4.2)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a4_2_frontier.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"Training {len(W_DOSE_VALUES)} controllers across the dose penalty range.")
    print(f"{TOTAL_TIMESTEPS:,} steps each, roughly a minute per controller.\n")

    agents = [train_one(w) for w in W_DOSE_VALUES]
    print()

    # Baselines on the same held-out patients.
    test_env = TumourImmuneEnv(split="test", include_params=True, seed=1)
    n_test = len(test_env.patient_ids)

    baselines = []
    for dose, label in [(0.0, "no treatment"), (0.2, "constant 0.2"),
                        (0.3, "constant 0.3"), (0.5, "constant 0.5"),
                        (0.7, "constant 0.7"), (V_MAX, "maximum dose")]:
        b = evaluate_policy(test_env, lambda obs, d=dose: d, n_test, label)
        baselines.append(b)
    baselines.append(
        evaluate_policy(test_env, lambda obs: 0.5 if obs[1] > 0.10 else 0.0,
                        n_test, "treat until T < 0.10")
    )

    print(f"Results on {n_test} held-out test patients.\n")
    print(f"{'policy':<26} {'controlled':>11} {'95 % interval':>18} {'mean dose':>11}")
    print("-" * 70)
    for b in baselines:
        successes = round(b["control_rate"] * n_test / 100.0)
        lo, hi = binomial_interval(successes, n_test)
        print(f"{b['label']:<26} {b['control_rate']:>10.1f}% "
              f"{f'[{lo:.1f}, {hi:.1f}]':>18} {b['mean_dose']:>11.2f}")
    print()
    for a in agents:
        t = a["test"]
        lo, hi = binomial_interval(t["controlled"], t["n"])
        name = f"learned, W={a['w_dose']:.2f}"
        interval = f"[{lo:.1f}, {hi:.1f}]"
        print(f"{name:<26} {t['control_rate']:>10.1f}% "
              f"{interval:>18} {t['mean_dose']:>11.2f}")
    print()

    # The dominance check, stated conservatively.
    print("Dominance check: for each learned controller, the cheapest constant-dose")
    print("baseline achieving at least the same control rate.\n")

    any_dominant = False
    for a in agents:
        rate, dose = a["test"]["control_rate"], a["test"]["mean_dose"]
        matching = [b for b in baselines if b["control_rate"] >= rate - 1e-9]
        if not matching:
            print(f"  W={a['w_dose']:.2f}: control rate {rate:.1f} % exceeds every baseline.")
            any_dominant = True
            continue
        cheapest = min(matching, key=lambda b: b["mean_dose"])
        if cheapest["mean_dose"] <= 0:
            print(f"  W={a['w_dose']:.2f}: matched only by no treatment at {rate:.1f} %.")
            continue
        saving = 100.0 * (1 - dose / cheapest["mean_dose"])
        verdict = "better" if saving > 0 else "worse"
        if saving > 0:
            any_dominant = True
        print(f"  W={a['w_dose']:.2f}: {rate:5.1f} % at dose {dose:6.2f}   "
              f"vs {cheapest['label']} at dose {cheapest['mean_dose']:6.2f}   "
              f"{saving:+6.1f} % drug ({verdict})")
    print()

    if any_dominant:
        print("The learned controllers reach the same outcomes on less drug at one or")
        print("more operating points. Report the frontier rather than a single point:")
        print("it is the pattern across settings that supports the claim, not any")
        print("individual comparison on 46 patients.")
    else:
        print("No learned controller improved on the baselines. Worth checking whether")
        print("training length or the reward shaping is the limitation.")
    print()

    fig_path = plot_frontier(agents, baselines)

    summary = {
        "n_test_patients": n_test,
        "total_timesteps": TOTAL_TIMESTEPS,
        "baselines": [
            {"label": b["label"], "control_rate": b["control_rate"],
             "mean_dose": b["mean_dose"]} for b in baselines
        ],
        "learned": [
            {"w_dose": a["w_dose"],
             "control_rate": a["test"]["control_rate"],
             "mean_dose": a["test"]["mean_dose"],
             "val_control_rate": a["val"]["control_rate"]} for a in agents
        ],
    }
    report_path = os.path.join(REPORT_DIR, "a4_2_frontier.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"  {fig_path}")
    print(f"  {report_path}")
