"""
Train the reinforcement learning controller (step A4.2).

The task
--------
Steer structurally treatable patients from tumour escape to immune control,
using less cumulative drug than a constant-dose schedule.

Diagnosis established that the drug is far too weak to kill a tumour outright:
its maximum kill rate is around 0.19 per unit time against a tumour growth
rate near 1.5. So the controller cannot win by force. It must push each
patient across their own separatrix and then stop, letting the immune system
finish. Personal separatrices range from 0.021 to 0.261 across the cohort,
which is why a single fixed threshold cannot work and state feedback can.

Baselines to beat (measured in A4.1, on training patients):

    policy                  control rate    mean dose
    no treatment                   0.0 %         0.00
    constant v = 0.3              13.0 %        30.00
    constant v = 0.5              23.1 %        50.00
    maximum dose                  37.5 %       100.00
    treat until T < 0.10          21.3 %        43.12

The claim being tested is relative, not absolute: a higher control rate than
the best constant-dose baseline, at lower cumulative dose. An absolute target
would be arbitrary, since the true achievable ceiling is not known.

Method
------
PPO from stable-baselines3, acting on the neural surrogate. Training uses the
training split; model selection uses validation patients; the final number
comes from test patients seen at no point during training.

Observations are normalised because the patient parameters span very
different scales (rho is around 0.01 while r1 is around 1.5), and an
unnormalised policy network would effectively ignore the small ones.

Run with:
    python src/train_controller.py
"""

import os
import time
import numpy as np
import torch

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import (
    TumourImmuneEnv, run_episode, evaluate_policy,
    V_MAX, N_STEPS, DT, N_FAIL, T_CONTROLLED,
)


MODEL_DIR = "models"
FIGURE_DIR = "figures"
REPORT_DIR = "reports"

RANDOM_SEED = 0

N_ENVS = 8              # parallel environments, for sample throughput
TOTAL_TIMESTEPS = 400_000
EVAL_EVERY = 25_000

# Keeping torch single-threaded avoids threads fighting each other when
# several environments each run a small network on CPU.
torch.set_num_threads(1)


# ----------------------------------------------------------------------
# Environment construction
# ----------------------------------------------------------------------

def make_env(split, seed, rank=0):
    """Build one environment instance, seeded so runs are reproducible."""
    def _init():
        env = TumourImmuneEnv(split=split, include_params=True, seed=seed + rank)
        return env
    return _init


def build_vec_env(split, seed, n_envs=1, norm_stats=None, training=True):
    """
    Build a vectorised, observation-normalised environment.

    Normalisation statistics are learned during training and then frozen for
    evaluation, so evaluation cannot benefit from statistics computed on the
    patients it is being scored against.
    """
    venv = DummyVecEnv([make_env(split, seed, i) for i in range(n_envs)])

    if norm_stats is None:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        venv = VecNormalize.load(norm_stats, venv)

    venv.training = training
    venv.norm_reward = False
    return venv


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_agent(model, split, norm_stats_path, label="agent"):
    """
    Run the trained policy over every patient in a split.

    Actions are taken deterministically, so the reported number reflects the
    learned policy rather than exploration noise.
    """
    venv = build_vec_env(split, seed=RANDOM_SEED + 999, n_envs=1,
                         norm_stats=norm_stats_path, training=False)
    raw_env = venv.venv.envs[0]
    n_patients = len(raw_env.patient_ids)

    results = []
    for index in range(n_patients):
        obs = venv.reset()
        # Pin the patient so every one is evaluated exactly once.
        raw_obs, _info = raw_env.reset(options={"patient_index": index})
        obs = venv.normalize_obs(raw_obs.reshape(1, -1))

        done = False
        total_reward = 0.0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            raw_obs, reward, terminated, truncated, info = raw_env.step(action[0])
            obs = venv.normalize_obs(raw_obs.reshape(1, -1))
            total_reward += reward
            done = terminated or truncated

        outcome = "host failure" if info["min_N"] < N_FAIL else (
            "controlled" if info["tumour"] < T_CONTROLLED else "escape"
        )
        results.append({
            "patient_id": info["patient_id"],
            "outcome": outcome,
            "final_T": info["tumour"],
            "min_N": info["min_N"],
            "total_dose": info["total_dose"],
            "reward": total_reward,
        })

    venv.close()

    controlled = sum(1 for r in results if r["outcome"] == "controlled")
    failures = sum(1 for r in results if r["outcome"] == "host failure")

    return {
        "label": label,
        "n": len(results),
        "control_rate": 100.0 * controlled / len(results),
        "failure_rate": 100.0 * failures / len(results),
        "mean_dose": float(np.mean([r["total_dose"] for r in results])),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "results": results,
    }


class ValidationCallback(BaseCallback):
    """
    Periodically score the policy on validation patients and keep the best.

    Selection uses control rate first, with mean dose breaking ties. Training
    reward alone would be a poor criterion, since the reward mixes tumour
    burden and dose into a single number that does not directly express the
    claim being made.
    """

    def __init__(self, norm_stats_path, eval_every, verbose=0):
        super().__init__(verbose)
        self.norm_stats_path = norm_stats_path
        self.eval_every = eval_every
        self.history = []
        self.best_score = (-1.0, float("inf"))
        self.best_path = os.path.join(MODEL_DIR, "controller_ppo_best.zip")

    def _on_step(self):
        if self.n_calls % self.eval_every != 0:
            return True

        # Save current normalisation statistics so evaluation matches training.
        self.model.get_vec_normalize_env().save(self.norm_stats_path)

        scores = evaluate_agent(self.model, "val", self.norm_stats_path, "val")
        self.history.append({
            "timesteps": self.num_timesteps,
            "control_rate": scores["control_rate"],
            "mean_dose": scores["mean_dose"],
            "mean_reward": scores["mean_reward"],
        })

        print(f"    {self.num_timesteps:>7,} steps   "
              f"validation control {scores['control_rate']:5.1f} %   "
              f"mean dose {scores['mean_dose']:6.2f}")

        # Higher control rate wins; at equal control rate, lower dose wins.
        score = (scores["control_rate"], -scores["mean_dose"])
        best = (self.best_score[0], -self.best_score[1])
        if score > best:
            self.best_score = (scores["control_rate"], scores["mean_dose"])
            self.model.save(self.best_path)
            print(f"              new best, saved")

        return True


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_results(history, agent_scores, baselines, agent_results):
    """Learning curve, trade-off frontier, and dosing behaviour."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Learning curve on validation patients.
    ax = axes[0]
    if history:
        steps = [h["timesteps"] for h in history]
        ax.plot(steps, [h["control_rate"] for h in history],
                "o-", color="#0f4c5c", label="control rate")
        ax.set_xlabel("training steps")
        ax.set_ylabel("validation control rate (%)")
        ax2 = ax.twinx()
        ax2.plot(steps, [h["mean_dose"] for h in history],
                 "s--", color="#a4243b", alpha=0.7, label="mean dose")
        ax2.set_ylabel("mean cumulative dose")
    ax.set_title("Learning progress")

    # The trade-off frontier: control rate against drug used.
    ax = axes[1]
    for b in baselines:
        ax.scatter(b["mean_dose"], b["control_rate"], s=70,
                   color="#888888", zorder=3)
        ax.annotate(b["label"], (b["mean_dose"], b["control_rate"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.scatter(agent_scores["mean_dose"], agent_scores["control_rate"],
               s=140, marker="*", color="#a4243b", zorder=4,
               edgecolor="white", linewidth=1.2)
    ax.annotate("learned controller",
                (agent_scores["mean_dose"], agent_scores["control_rate"]),
                textcoords="offset points", xytext=(8, -12),
                fontsize=8, color="#a4243b", fontweight="bold")
    ax.set_xlabel("mean cumulative dose")
    ax.set_ylabel("control rate (%)")
    ax.set_title("Trade-off: outcome against drug used\n(up and to the left is better)")

    # Distribution of drug used per patient.
    ax = axes[2]
    doses = [r["total_dose"] for r in agent_results]
    controlled = [r["total_dose"] for r in agent_results if r["outcome"] == "controlled"]
    ax.hist(doses, bins=25, color="#0f4c5c", alpha=0.6, label="all patients")
    if controlled:
        ax.hist(controlled, bins=25, color="#1b7a5a", alpha=0.75,
                label="successfully controlled")
    ax.axvline(100.0, color="#a4243b", linestyle="--", linewidth=1.3,
               label="maximum dose baseline")
    ax.set_xlabel("cumulative dose for this patient")
    ax.set_ylabel("patients")
    ax.set_title("Dose varies by patient")
    ax.legend(fontsize=7)

    fig.suptitle("Learned controller (A4.2)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a4_2_controller.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    norm_stats_path = os.path.join(MODEL_DIR, "controller_vecnormalize.pkl")

    print("Building training environments...")
    train_venv = build_vec_env("train", RANDOM_SEED, n_envs=N_ENVS, training=True)
    n_train_patients = len(train_venv.venv.envs[0].patient_ids)
    print(f"  {N_ENVS} parallel environments, {n_train_patients} treatable "
          f"training patients")
    print(f"  episode length {N_STEPS} steps of DT = {DT}\n")

    model = PPO(
        "MlpPolicy",
        train_venv,
        learning_rate=3e-4,
        n_steps=256,              # per environment, so 2048 samples per update
        batch_size=256,
        n_epochs=10,
        gamma=0.995,              # long horizon, so future reward matters
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,           # mild exploration pressure
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=RANDOM_SEED,
        verbose=0,
    )

    callback = ValidationCallback(norm_stats_path, eval_every=EVAL_EVERY // N_ENVS)

    print(f"Training PPO for {TOTAL_TIMESTEPS:,} steps...")
    print("  (validation scores are printed periodically)\n")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - start

    print(f"\n  training finished in {elapsed / 60:.1f} minutes\n")

    # Reload the best checkpoint by validation performance, not the last one.
    train_venv.save(norm_stats_path)
    best_path = callback.best_path
    if os.path.exists(best_path):
        model = PPO.load(best_path)
        print(f"Loaded best checkpoint: validation control "
              f"{callback.best_score[0]:.1f} %, dose {callback.best_score[1]:.2f}\n")

    # ------------------------------------------------------------------
    # Final evaluation on held-out test patients.
    # ------------------------------------------------------------------
    print("Evaluating on held-out test patients...\n")

    agent_scores = evaluate_agent(model, "test", norm_stats_path, "learned controller")

    test_env = TumourImmuneEnv(split="test", include_params=True, seed=1)
    n_test = len(test_env.patient_ids)

    baselines = [
        evaluate_policy(test_env, lambda obs: 0.0, n_test, "no treatment"),
        evaluate_policy(test_env, lambda obs: 0.3, n_test, "constant v = 0.3"),
        evaluate_policy(test_env, lambda obs: 0.5, n_test, "constant v = 0.5"),
        evaluate_policy(test_env, lambda obs: V_MAX, n_test, "maximum dose"),
        evaluate_policy(test_env, lambda obs: 0.5 if obs[1] > 0.10 else 0.0,
                        n_test, "treat until T < 0.10"),
    ]

    print(f"Results on {n_test} unseen test patients:\n")
    print(f"{'policy':<26} {'controlled':>11} {'mean dose':>11} {'mean reward':>12}")
    print("-" * 62)
    for b in baselines:
        print(f"{b['label']:<26} {b['control_rate']:>10.1f}% "
              f"{b['mean_dose']:>11.2f} {b['mean_reward']:>12.2f}")
    print(f"{agent_scores['label']:<26} {agent_scores['control_rate']:>10.1f}% "
          f"{agent_scores['mean_dose']:>11.2f} {agent_scores['mean_reward']:>12.2f}")
    print()

    # The claim, stated against the strongest baseline.
    best_baseline = max(baselines, key=lambda b: b["control_rate"])
    print(f"Best baseline by control rate: {best_baseline['label']} "
          f"({best_baseline['control_rate']:.1f} % at dose {best_baseline['mean_dose']:.2f})")
    print()

    better_control = agent_scores["control_rate"] >= best_baseline["control_rate"]
    lower_dose = agent_scores["mean_dose"] < best_baseline["mean_dose"]

    if better_control and lower_dose:
        saving = 100.0 * (1 - agent_scores["mean_dose"] / best_baseline["mean_dose"])
        print(f"CLAIM SUPPORTED: the controller matches or exceeds the best")
        print(f"baseline's control rate using {saving:.1f} % less drug.")
    elif better_control:
        print("The controller achieves a higher control rate, but not at lower dose.")
    elif lower_dose:
        print("The controller uses less drug, but does not match the best control rate.")
        print("Report the trade-off honestly rather than selecting a favourable baseline.")
    else:
        print("The controller does not yet beat the baselines. Consider training")
        print("longer, or revisiting the dose penalty W_DOSE in env.py.")
    print()

    # ------------------------------------------------------------------
    fig_path = plot_results(callback.history, agent_scores, baselines,
                            agent_scores["results"])

    report_path = os.path.join(REPORT_DIR, "a4_2_controller.md")
    with open(report_path, "w") as f:
        f.write("# Learned Controller (A4.2)\n\n")
        f.write(f"PPO trained for {TOTAL_TIMESTEPS:,} steps on "
                f"{n_train_patients} treatable training patients.\n")
        f.write(f"Evaluated on {n_test} held-out test patients.\n\n")
        f.write("| policy | control rate | mean dose | mean reward |\n")
        f.write("|---|---|---|---|\n")
        for b in baselines:
            f.write(f"| {b['label']} | {b['control_rate']:.1f} % | "
                    f"{b['mean_dose']:.2f} | {b['mean_reward']:.2f} |\n")
        f.write(f"| **{agent_scores['label']}** | "
                f"**{agent_scores['control_rate']:.1f} %** | "
                f"**{agent_scores['mean_dose']:.2f}** | "
                f"**{agent_scores['mean_reward']:.2f}** |\n\n")
        f.write("## Notes\n\n")
        f.write("- Patients are structurally treatable only: a healthy attractor "
                "exists for them. Around 70 percent of patients needing rescue are "
                "monostable and cannot be helped by any dose.\n")
        f.write("- Host failure did not occur under any policy in this population, "
                "so the safety penalty in the reward was never triggered. The "
                "problem is control rate against dose economy.\n")

    print("Saved:")
    print(f"  {fig_path}")
    print(f"  {report_path}")
    print(f"  {best_path}")
