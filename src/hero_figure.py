"""
Demonstrate steering (step A4.4).

The project's claim in one image: a patient who would progress to a chronic
diseased state under no treatment is instead driven to tumour clearance by
the learned controller, using a fraction of the drug a constant schedule
would need.

The figure also answers a question nothing in the reward function specified:
does the controller learn to STOP once the patient has crossed their own
separatrix? The drug is far too weak to kill a tumour outright, so the only
winning strategy is to push a patient into the healthy basin and let the
immune system finish. If the controller discovered that unaided, it is an
emergent result rather than a designed one.

Run with:
    python src/hero_figure.py
"""

import os
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import Params
from simulate import simulate, classify
from control import simulate_controlled
from cohort import make_params
from env import TumourImmuneEnv, V_MAX, N_STEPS, DT, T_CONTROLLED


DATA_DIR = "data"
MODEL_DIR = "models"
FIGURE_DIR = "figures"

# W = 0.60 is the most economical controller, and the one that strictly
# dominates the constant 0.7 baseline: higher control rate on 60 percent
# less drug.
W_DOSE = 0.60
TAG = f"w{W_DOSE:.2f}".replace(".", "")

COLOUR_UNTREATED = "#a4243b"
COLOUR_CONTROLLED = "#1b7a5a"
COLOUR_DOSE = "#0f4c5c"
COLOUR_ACCENT = "#c78c3c"

torch.set_num_threads(1)


def load_controller():
    """Load the trained controller and its observation normalisation."""
    model_path = os.path.join(MODEL_DIR, f"frontier_{TAG}.zip")
    norm_path = os.path.join(MODEL_DIR, f"frontier_{TAG}_vecnorm.pkl")

    for p in (model_path, norm_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run 'python src/frontier.py' first.")

    model = PPO.load(model_path)

    def _init():
        return TumourImmuneEnv(split="test", include_params=True,
                               seed=99, w_dose=W_DOSE)

    venv = VecNormalize.load(norm_path, DummyVecEnv([_init]))
    venv.training = False
    venv.norm_reward = False

    return model, venv, venv.venv.envs[0]


def run_controlled(model, venv, raw_env, patient_index):
    """Run one patient under the learned controller, recording the trajectory."""
    venv.reset()
    obs_raw, _ = raw_env.reset(options={"patient_index": patient_index})
    obs = venv.normalize_obs(obs_raw.reshape(1, -1))

    states = [raw_env.state.copy()]
    doses = []
    done, info = False, {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs_raw, reward, terminated, truncated, info = raw_env.step(action[0])
        obs = venv.normalize_obs(obs_raw.reshape(1, -1))

        states.append(raw_env.state.copy())
        doses.append(float(np.clip(action[0][0], 0.0, V_MAX)))
        done = terminated or truncated

    return np.array(states), np.array(doses), info


def personal_separatrix(params, y0, low=0.005, high=0.60, tolerance=1e-3):
    """
    Find this specific patient's separatrix by bisection.

    Each patient has their own boundary between the healthy and diseased
    basins. Across the cohort these range from about 0.02 to 0.26, which is
    why a single fixed treatment threshold cannot work.
    """
    def escapes(T0):
        probe = y0.copy()
        probe[1] = T0
        t, Y = simulate(probe, params=params, v=0.0, t_end=200.0, n_points=500)
        return classify(Y[1], Y[0]) != "controlled"

    if escapes(low) or not escapes(high):
        return None

    iterations = 0
    while (high - low) > tolerance and iterations < 40:
        mid = 0.5 * (low + high)
        if escapes(mid):
            high = mid
        else:
            low = mid
        iterations += 1

    return 0.5 * (low + high)


def analyse_all_patients(model, venv, raw_env):
    """
    Run every test patient and measure the controller's dosing behaviour.

    A single patient cannot tell us whether the controller learned a stopping
    rule. Dose requirements vary enormously across patients, so the question
    has to be asked of the whole population.

    For each rescued patient this records when the tumour crosses their own
    separatrix and when dosing effectively ends, so the two can be compared.
    """
    cohort = np.load(os.path.join(DATA_DIR, "cohort.npz"), allow_pickle=True)
    param_names = [str(x) for x in cohort["param_names"]]

    records = []
    for index in range(len(raw_env.patient_ids)):
        states, doses, info = run_controlled(model, venv, raw_env, index)
        pid = int(raw_env.patient_ids[index])

        params = make_params({
            name: float(cohort["params"][pid][k])
            for k, name in enumerate(param_names)
        })
        y0 = cohort["y0"][pid].astype(float)
        sep = personal_separatrix(params, y0)

        rescued = info["tumour"] < T_CONTROLLED

        # When the tumour first drops below this patient's separatrix.
        cross_time = None
        if sep is not None:
            below = np.flatnonzero(states[:, 1] < sep)
            if len(below):
                cross_time = float(below[0] * DT)

        # When dosing effectively ends.
        dosing = np.flatnonzero(doses > 0.02)
        stop_time = float((dosing[-1] + 1) * DT) if len(dosing) else 0.0

        records.append({
            "index": index,
            "patient_id": pid,
            "params": params,
            "y0": y0,
            "separatrix": sep,
            "states": states,
            "doses": doses,
            "total_dose": float(info["total_dose"]),
            "rescued": rescued,
            "cross_time": cross_time,
            "stop_time": stop_time,
            "difficulty": (y0[1] - sep) if sep is not None else None,
        })

    return records


def pick_patient(records):
    """
    Choose a REPRESENTATIVE rescued patient, not an extreme one.

    An earlier version selected the patient furthest above their separatrix,
    which reliably found the hardest case in the set: one requiring more than
    three times the average dose. That misrepresents typical behaviour.

    Selecting the median-dose rescued patient shows what the controller
    usually does.
    """
    rescued = [r for r in records
               if r["rescued"] and r["separatrix"] is not None
               and r["cross_time"] is not None]

    if not rescued:
        raise RuntimeError("No rescued patient with a well-defined separatrix.")

    rescued.sort(key=lambda r: r["total_dose"])
    return rescued[len(rescued) // 2]


def make_figure(patient):
    """Build the three-panel demonstration figure."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    params = patient["params"]
    y0 = patient["y0"]
    sep = patient["separatrix"]
    ctrl_states = patient["states"]
    ctrl_doses = patient["doses"]

    time = np.arange(len(ctrl_states)) * DT
    dose_time = np.arange(len(ctrl_doses)) * DT

    # Untreated trajectory from the mechanistic model.
    t_un, Y_un = simulate(y0, params=params, v=0.0,
                          t_end=N_STEPS * DT, n_points=len(ctrl_states))

    # When does the tumour cross the separatrix, and when does dosing stop?
    below = np.flatnonzero(ctrl_states[:, 1] < sep)
    cross_time = below[0] * DT if len(below) else None

    dosing = np.flatnonzero(ctrl_doses > 0.02)
    stop_time = (dosing[-1] + 1) * DT if len(dosing) else 0.0

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.28)

    # ---- Panel 1: tumour over time ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_un, Y_un[1], color=COLOUR_UNTREATED, linewidth=2.4,
            label="untreated")
    ax.plot(time, ctrl_states[:, 1], color=COLOUR_CONTROLLED, linewidth=2.4,
            label="learned controller")
    ax.axhline(sep, color=COLOUR_ACCENT, linestyle="--", linewidth=1.4)
    ax.text(time[-1] * 0.62, sep + 0.012,
            f"this patient's separatrix, T = {sep:.3f}",
            fontsize=8, color=COLOUR_ACCENT)

    if cross_time is not None:
        ax.axvline(cross_time, color=COLOUR_ACCENT, linestyle=":", linewidth=1.2)

    ax.set_ylabel("tumour population  T")
    ax.set_title("The tumour is driven to clearance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)

    # ---- Panel 1 lower: applied dose ----
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(dose_time, 0, ctrl_doses, color=COLOUR_DOSE, alpha=0.75, step="post")
    ax.axhline(0.7, color="#999999", linestyle="--", linewidth=1.0)
    ax.text(time[-1] * 0.55, 0.72, "constant 0.7 baseline", fontsize=7, color="#777777")
    if cross_time is not None:
        ax.axvline(cross_time, color=COLOUR_ACCENT, linestyle=":", linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel("dose  v")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"treatment stops at t = {stop_time:.0f}", fontsize=9)
    ax.grid(alpha=0.25)

    # ---- Panel 2: healthy tissue ----
    ax = fig.add_subplot(gs[:, 1])
    ax.plot(t_un, Y_un[0], color=COLOUR_UNTREATED, linewidth=2.4, label="untreated")
    ax.plot(time, ctrl_states[:, 0], color=COLOUR_CONTROLLED, linewidth=2.4,
            label="learned controller")
    ax.plot(t_un, Y_un[2], color=COLOUR_UNTREATED, linewidth=1.2,
            linestyle="--", alpha=0.6, label="untreated immune")
    ax.plot(time, ctrl_states[:, 2], color=COLOUR_CONTROLLED, linewidth=1.2,
            linestyle="--", alpha=0.6, label="controlled immune")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("Healthy tissue recovers, immunity is preserved",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ---- Panel 3: phase plane ----
    ax = fig.add_subplot(gs[:, 2])
    ax.plot(Y_un[1], Y_un[0], color=COLOUR_UNTREATED, linewidth=2.2,
            label="untreated path")
    ax.plot(ctrl_states[:, 1], ctrl_states[:, 0], color=COLOUR_CONTROLLED,
            linewidth=2.2, label="controlled path")

    ax.plot(y0[1], y0[0], "o", color="#333333", markersize=8,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate("start", (y0[1], y0[0]), textcoords="offset points",
                xytext=(8, 6), fontsize=8)

    ax.plot(Y_un[1][-1], Y_un[0][-1], "X", color=COLOUR_UNTREATED, markersize=13,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate("diseased state", (Y_un[1][-1], Y_un[0][-1]),
                textcoords="offset points", xytext=(-14, -20),
                fontsize=8, color=COLOUR_UNTREATED)

    ax.plot(ctrl_states[-1, 1], ctrl_states[-1, 0], "*", color=COLOUR_CONTROLLED,
            markersize=20, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate("healthy state", (ctrl_states[-1, 1], ctrl_states[-1, 0]),
                textcoords="offset points", xytext=(10, -6),
                fontsize=8, color=COLOUR_CONTROLLED)

    ax.axvline(sep, color=COLOUR_ACCENT, linestyle="--", linewidth=1.3)
    ax.set_xlabel("tumour population  T")
    ax.set_ylabel("healthy cell population  N")
    ax.set_title("Steering between attractors", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Steering a virtual patient from tumour escape to immune control "
        f"(patient {patient['patient_id']}, A4.4)",
        fontsize=13, fontweight="bold",
    )

    path = os.path.join(FIGURE_DIR, "a4_4_steering.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path, cross_time, stop_time


if __name__ == "__main__":
    print(f"Loading controller W = {W_DOSE}...\n")
    model, venv, raw_env = load_controller()

    print("Running every test patient to measure dosing behaviour...")
    records = analyse_all_patients(model, venv, raw_env)
    rescued = [r for r in records if r["rescued"]]
    print(f"  {len(rescued)} of {len(records)} patients rescued\n")

    # How much does the dose vary with how hard the patient is?
    doses = [r["total_dose"] for r in rescued]
    print("Dose adaptation across rescued patients:")
    print(f"  minimum  {min(doses):6.2f}")
    print(f"  median   {np.median(doses):6.2f}")
    print(f"  mean     {np.mean(doses):6.2f}")
    print(f"  maximum  {max(doses):6.2f}")
    print()

    hard = [r for r in rescued if r["difficulty"] is not None]
    if len(hard) > 3:
        difficulty = np.array([r["difficulty"] for r in hard])
        spend = np.array([r["total_dose"] for r in hard])
        corr = float(np.corrcoef(difficulty, spend)[0, 1])
        print(f"  correlation between patient difficulty and drug used: {corr:+.2f}")
        if corr > 0.4:
            print("  The controller spends more on patients who are further from")
            print("  their healthy basin. This is per-patient adaptation, and it is")
            print("  what a fixed schedule cannot do.")
        print()

    # Does a stopping rule emerge, across the population rather than in one case?
    with_cross = [r for r in rescued
                  if r["cross_time"] is not None and r["separatrix"] is not None]
    if with_cross:
        lags = [r["stop_time"] - r["cross_time"] for r in with_cross]
        stops_early = sum(1 for lag in lags if lag <= 15.0)
        print("Stopping behaviour, measured across all rescued patients:")
        print(f"  median lag between crossing the separatrix and stopping: "
              f"{np.median(lags):+.1f} time units")
        print(f"  stopped within 15 units of crossing: {stops_early} / {len(with_cross)}")
        print()
        if np.median(lags) <= 15.0:
            print("  The controller typically stops shortly after pushing a patient")
            print("  into the healthy basin, leaving the immune system to finish.")
            print("  Nothing in the reward specified this; it emerged from the dose")
            print("  penalty alone.")
        else:
            print("  The controller typically keeps dosing past the crossing. It")
            print("  found an economical policy without discovering the stopping")
            print("  rule, so dose economy could still be improved.")
        print()

    patient = pick_patient(records)
    print(f"Illustrating a representative patient (median dose among rescued):")
    print(f"  patient {patient['patient_id']}")
    print(f"  starting tumour   {patient['y0'][1]:.4f}")
    print(f"  their separatrix  {patient['separatrix']:.4f}")
    print(f"  drug used         {patient['total_dose']:.2f}  "
          f"(cohort mean {np.mean(doses):.2f})")
    print()

    path, cross_time, stop_time = make_figure(patient)
    print(f"Saved: {path}")
