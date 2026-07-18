"""
Gymnasium control environment wrapping the neural surrogate (step A4.1).

Why the surrogate rather than the ODE
-------------------------------------
Reinforcement learning needs millions of environment steps. Integrating the
mechanistic model that many times would take days. The surrogate produces a
step in a single network evaluation, which is what makes Phase 4 feasible at
all. This is the practical payoff of Phase 3.

Reward design
-------------
Each step the agent is penalised for tumour burden and for drug administered:

    reward = -(W_TUMOUR * T + W_DOSE * v) * DT

The dose penalty is the important half. Without it the agent would simply
administer the maximum dose forever, which does control the tumour but is
exactly the wasteful behaviour the project sets out to improve on. With it,
the agent must find the economical solution, which is what makes the
"less cumulative drug" claim testable rather than assumed.

Two terminal signals sharpen this:
  - a large penalty if healthy tissue collapses (the A0.3 host failure rule)
  - a bonus for reaching and holding tumour control

Observation
-----------
By default the agent observes the state AND the patient's parameters. This
matches the digital twin premise: the model is calibrated to a known
individual. Set include_params=False for the harder partial-information
problem, where the controller must infer the patient from their response.

Patient sampling
----------------
Training environments draw from the TRAINING split only, evaluation
environments from the test split. The patient-level separation established
in A2.3 is preserved here, so a controller is never evaluated on a patient
it practised on.

Run with:
    python src/env.py
"""

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from train_surrogate import Surrogate, Normaliser
from cohort import NEEDS_RESCUE


DATA_DIR = "data"
MODEL_DIR = "models"

DT = 0.5
T_END = 100.0
N_STEPS = int(T_END / DT)     # 200 steps per episode

V_MAX = 1.0

# Reward weights. Chosen so tumour burden dominates, with the dose penalty
# large enough to matter but not so large that the agent refuses to treat.
W_TUMOUR = 1.0
W_DOSE = 0.15

# Terminal signals, from the A0.3 metric definitions.
N_FAIL = 0.20
HOST_FAILURE_PENALTY = 50.0
T_CONTROLLED = 0.05
CONTROL_BONUS = 25.0


def load_surrogate(device=torch.device("cpu")):
    """Load the trained surrogate and its normalisation statistics."""
    path = os.path.join(MODEL_DIR, "surrogate_mlp.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run train_surrogate.py first.")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Surrogate(
        ckpt["n_inputs"], ckpt["n_outputs"],
        hidden=ckpt["hidden"], n_layers=ckpt["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, Normaliser.from_state_dict(ckpt["x_norm"]), \
        Normaliser.from_state_dict(ckpt["y_norm"])


def load_patients(split="train", treatable_only=True):
    """
    Load the patient pool for one split.

    The controller trains on patients who both need rescue AND can be
    rescued. Diagnosis established that the drug is far too weak to kill a
    tumour directly; it only pushes a patient across their separatrix so the
    immune system can finish. Patients with no healthy attractor are
    monostable and cannot be helped by any dose, so including them would
    measure an impossible task rather than a hard one.

    The monostable group is reported separately as a structural finding.
    """
    cohort = np.load(os.path.join(DATA_DIR, "cohort.npz"), allow_pickle=True)
    splits = np.load(os.path.join(DATA_DIR, "splits.npz"), allow_pickle=True)

    key = {"train": "train_patients", "val": "val_patients", "test": "test_patients"}[split]
    ids = splits[key]

    if treatable_only:
        if "treatable" in cohort.files:
            treatable = cohort["treatable"]
            ids = np.array([i for i in ids if treatable[i]])
        else:
            outcomes = np.array([str(x) for x in cohort["outcomes"]])
            ids = np.array([i for i in ids if outcomes[i] in NEEDS_RESCUE])

    if len(ids) == 0:
        raise ValueError(f"No patients available in split '{split}'.")

    return ids, cohort["params"][ids].astype(np.float32), \
        cohort["y0"][ids].astype(np.float32)


class TumourImmuneEnv(gym.Env):
    """
    A treatment episode for one virtual patient.

    Observation : [N, T, I, u] plus 8 patient parameters (if included)
    Action      : a single dose rate in [0, V_MAX]
    Episode     : 200 steps of DT = 0.5, or early termination on host failure
    """

    metadata = {"render_modes": []}

    def __init__(self, split="train", include_params=True, seed=None,
                 device=torch.device("cpu"), w_dose=None):
        super().__init__()


        # The dose penalty weight sets where on the trade-off between
        # control rate and drug economy this controller sits. Sweeping it
        # traces out a frontier rather than a single operating point.
        self.w_dose = W_DOSE if w_dose is None else float(w_dose)
        self.device = device
        self.include_params = include_params
        self.split = split

        self.model, self.x_norm, self.y_norm = load_surrogate(device)
        self.patient_ids, self.patient_params, self.patient_y0 = load_patients(split)

        self.action_space = spaces.Box(low=0.0, high=V_MAX, shape=(1,), dtype=np.float32)

        # Generous bounds. The surrogate can briefly leave physical ranges, and
        # clipping observations would hide that rather than handling it.
        n_obs = 4 + (self.patient_params.shape[1] if include_params else 0)
        self.observation_space = spaces.Box(
            low=-1.0, high=5.0, shape=(n_obs,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.state = None
        self.params = None
        self.step_count = 0
        self.total_dose = 0.0
        self.min_N = 1.0
        self.patient_index = None

    def _observation(self):
        if self.include_params:
            return np.concatenate([self.state, self.params]).astype(np.float32)
        return self.state.astype(np.float32)

    def reset(self, seed=None, options=None):
        """Start a new episode with a randomly chosen patient from this split."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # options={"patient_index": i} pins a specific patient, used for
        # reproducible evaluation.
        if options and "patient_index" in options:
            self.patient_index = int(options["patient_index"])
        else:
            self.patient_index = int(self._rng.integers(len(self.patient_ids)))

        self.state = self.patient_y0[self.patient_index].copy()
        self.params = self.patient_params[self.patient_index].copy()

        self.step_count = 0
        self.total_dose = 0.0
        self.min_N = float(self.state[0])

        return self._observation(), {"patient_id": int(self.patient_ids[self.patient_index])}

    def step(self, action):
        """Apply a dose for one timestep and advance the surrogate."""
        dose = float(np.clip(action[0], 0.0, V_MAX))

        # Advance the state using the surrogate.
        with torch.no_grad():
            x = torch.tensor(
                np.concatenate([self.state, [dose], self.params]),
                dtype=torch.float32,
            ).unsqueeze(0)
            delta = self.y_norm.decode(
                self.model(self.x_norm.encode(x).to(self.device)).cpu()
            ).squeeze(0).numpy()

        self.state = self.state + delta
        self.step_count += 1
        self.total_dose += dose * DT

        N, T = float(self.state[0]), float(self.state[1])
        self.min_N = min(self.min_N, N)

        # Running cost: tumour burden and drug used.
        reward = -(W_TUMOUR * T + self.w_dose * dose) * DT

        terminated = False
        truncated = False

        # Host failure ends the episode badly, per the A0.3 rule.
        if N < N_FAIL:
            reward -= HOST_FAILURE_PENALTY
            terminated = True

        elif self.step_count >= N_STEPS:
            truncated = True
            # Bonus only if the tumour is genuinely controlled at the end.
            if T < T_CONTROLLED:
                reward += CONTROL_BONUS

        info = {
            "patient_id": int(self.patient_ids[self.patient_index]),
            "tumour": T,
            "healthy": N,
            "dose": dose,
            "total_dose": self.total_dose,
            "min_N": self.min_N,
        }

        return self._observation(), float(reward), terminated, truncated, info


# ----------------------------------------------------------------------
# Baseline policies for comparison, reusing the A1.3 ideas
# ----------------------------------------------------------------------

def run_episode(env, policy_fn, patient_index=None, seed=None):
    """Run one full episode under a given policy and summarise it."""
    options = {"patient_index": patient_index} if patient_index is not None else None
    obs, info = env.reset(seed=seed, options=options)

    total_reward = 0.0
    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        total_reward += reward
        done = terminated or truncated

    outcome = "host failure" if info["min_N"] < N_FAIL else (
        "controlled" if info["tumour"] < T_CONTROLLED else "escape"
    )

    return {
        "reward": total_reward,
        "outcome": outcome,
        "final_T": info["tumour"],
        "min_N": info["min_N"],
        "total_dose": info["total_dose"],
        "patient_id": info["patient_id"],
    }


def evaluate_policy(env, policy_fn, n_patients, label):
    """Run a policy across a whole split and report aggregate metrics."""
    results = [run_episode(env, policy_fn, patient_index=i) for i in range(n_patients)]

    controlled = sum(1 for r in results if r["outcome"] == "controlled")
    failures = sum(1 for r in results if r["outcome"] == "host failure")
    mean_dose = np.mean([r["total_dose"] for r in results])
    mean_reward = np.mean([r["reward"] for r in results])

    return {
        "label": label,
        "control_rate": 100.0 * controlled / len(results),
        "failure_rate": 100.0 * failures / len(results),
        "mean_dose": mean_dose,
        "mean_reward": mean_reward,
        "results": results,
    }


if __name__ == "__main__":
    print("Building environment...\n")
    env = TumourImmuneEnv(split="train", include_params=True, seed=0)

    print(f"  observation space: {env.observation_space.shape}")
    print(f"  action space:      {env.action_space.shape}, "
          f"range [0, {V_MAX}]")
    print(f"  patients in split: {len(env.patient_ids)}")
    print(f"  episode length:    {N_STEPS} steps of DT = {DT}")
    print()

    # Check the environment satisfies the Gymnasium contract. Catching an
    # interface error now is far cheaper than debugging it inside a training run.
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("Gymnasium environment check passed.\n")
    except Exception as exc:
        print(f"Environment check reported: {exc}\n")

    # Baselines, so the RL agent in A4.2 has something honest to beat.
    n = len(env.patient_ids)

    baselines = [
        evaluate_policy(env, lambda obs: 0.0, n, "no treatment"),
        evaluate_policy(env, lambda obs: 0.3, n, "constant v = 0.3"),
        evaluate_policy(env, lambda obs: 0.5, n, "constant v = 0.5"),
        evaluate_policy(env, lambda obs: V_MAX, n, "maximum dose"),
        # The A1.3 feedback rule: treat until the tumour is well inside the
        # healthy basin, then stop. obs[1] is the tumour population.
        evaluate_policy(env, lambda obs: 0.5 if obs[1] > 0.10 else 0.0, n,
                        "treat until T < 0.10"),
    ]

    print(f"Baseline policies on {n} structurally treatable training patients:\n")
    print(f"{'policy':<24} {'controlled':>11} {'host fail':>10} "
          f"{'mean dose':>10} {'mean reward':>12}")
    print("-" * 70)
    for b in baselines:
        print(f"{b['label']:<24} {b['control_rate']:>10.1f}% {b['failure_rate']:>9.1f}% "
              f"{b['mean_dose']:>10.2f} {b['mean_reward']:>12.2f}")
    print()

    best = max(baselines, key=lambda b: b["mean_reward"])
    print(f"Best baseline by reward: {best['label']}")
    print(f"  control rate {best['control_rate']:.1f} %, "
          f"mean dose {best['mean_dose']:.2f}")
    print()
    print("These are the numbers the learned controller must beat in A4.2.")
    print("Beating the best of these on dose while matching control rate is")
    print("the project's headline claim.")
