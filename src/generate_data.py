"""
Generate surrogate training data (step A2.2).

The surrogate must learn how the drug changes the dynamics, not just what
happens when a patient is left alone. So each patient is simulated under
several randomised dosing schedules, and every transition is recorded.

What a transition is
--------------------
The surrogate does not learn continuous time. It learns a fixed-step map:

    given the current state and the dose applied, what is the state DT later?

Each recorded sample is therefore (state, dose, next state). Rolling that
map forward repeatedly reproduces a trajectory.

Choice of timestep
------------------
DT = 0.5 across a horizon of 100 gives 200 steps per trajectory. Larger
steps miss fast transients; smaller steps need more rollout steps and
accumulate more error over a full horizon.

Dosing schedules
----------------
Piecewise constant: a dose is drawn, held for HOLD time units, then redrawn.
Some segments are deliberately zero so the surrogate sees treatment starting,
stopping, and resuming. Every patient also gets one fully untreated run so
the natural dynamics are well covered.

Outputs
-------
    data/transitions.npz    the training data, regenerable so not committed
    figures/a2_2_data.png   coverage of the generated data

Run with:
    python src/generate_data.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Params
from control import simulate_controlled, V_MAX
from cohort import make_params, PARAMETER_RANGES


DATA_DIR = "data"
FIGURE_DIR = "figures"

RANDOM_SEED = 7

DT = 0.5            # timestep the surrogate will learn
T_END = 100.0       # horizon per trajectory
HOLD = 5.0          # how long each dose segment is held
N_SCHEDULES = 4     # randomised schedules per patient, plus one untreated run
P_ZERO = 0.30       # chance any given segment is a rest period


def random_schedule(rng, t_end=T_END, hold=HOLD, v_max=V_MAX, p_zero=P_ZERO):
    """
    Build one piecewise-constant dosing schedule.

    Returns a policy function v(t, y) -> dose, matching the interface from
    step A1.3, so it plugs straight into simulate_controlled.
    """
    n_segments = int(np.ceil(t_end / hold)) + 1

    # Draw a dose for each segment, with some segments forced to zero.
    doses = rng.uniform(0.0, v_max, size=n_segments)
    doses[rng.random(n_segments) < p_zero] = 0.0

    def policy(t, y):
        index = int(t // hold)
        index = min(index, n_segments - 1)
        return float(doses[index])

    return policy, doses


def zero_schedule():
    """The untreated policy, so natural dynamics are always represented."""
    def policy(t, y):
        return 0.0
    return policy, np.zeros(1)


def transitions_from_trajectory(t, Y, v_series):
    """
    Convert a trajectory into (state, dose, next state) samples.

    The trajectory is recorded on a fine grid for accuracy, then subsampled
    onto the DT grid the surrogate will actually use.
    """
    # Index step that corresponds to DT on this trajectory's time grid.
    dt_grid = t[1] - t[0]
    stride = max(1, int(round(DT / dt_grid)))

    idx = np.arange(0, len(t), stride)
    states = Y[:, idx].T            # shape (n_steps, 4)
    doses = v_series[idx]           # shape (n_steps,)

    # A transition needs a "next" state, so the final point is dropped.
    current = states[:-1]
    nxt = states[1:]
    applied = doses[:-1]

    return current, applied, nxt


def generate(cohort_path=os.path.join(DATA_DIR, "cohort.npz"), seed=RANDOM_SEED,
             verbose=True):
    """Simulate every patient under several schedules and collect transitions."""
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(
            f"{cohort_path} not found. Run 'python src/cohort.py' first."
        )

    cohort = np.load(cohort_path, allow_pickle=True)
    param_names = [str(x) for x in cohort["param_names"]]
    params_array = cohort["params"]
    y0_array = cohort["y0"]
    n_patients = len(y0_array)

    rng = np.random.default_rng(seed)

    all_states, all_doses, all_next = [], [], []
    all_patient_ids, all_params = [], []

    # Enough resolution that subsampling onto the DT grid is exact.
    n_points = int(T_END / (DT / 4.0)) + 1

    for patient_id in range(n_patients):
        patient_params = {
            name: float(params_array[patient_id, i])
            for i, name in enumerate(param_names)
        }
        p = make_params(patient_params)
        y0 = y0_array[patient_id]

        # One untreated run plus several randomised schedules.
        schedules = [zero_schedule()] + [
            random_schedule(rng) for _ in range(N_SCHEDULES)
        ]

        for policy, _doses in schedules:
            try:
                t, Y, v_series, _total = simulate_controlled(
                    y0, policy, params=p, t_end=T_END, n_points=n_points
                )
            except RuntimeError:
                # A stiff patient-schedule combination can fail to integrate.
                # Skipping it is honest; it is recorded in the summary count.
                continue

            current, applied, nxt = transitions_from_trajectory(t, Y, v_series)

            all_states.append(current)
            all_doses.append(applied)
            all_next.append(nxt)
            all_patient_ids.append(np.full(len(current), patient_id))
            all_params.append(np.tile(params_array[patient_id], (len(current), 1)))

        if verbose and (patient_id + 1) % 50 == 0:
            print(f"  simulated {patient_id + 1} / {n_patients} patients...")

    return {
        "states": np.concatenate(all_states),
        "doses": np.concatenate(all_doses),
        "next_states": np.concatenate(all_next),
        "patient_ids": np.concatenate(all_patient_ids),
        "patient_params": np.concatenate(all_params),
        "param_names": param_names,
        "dt": DT,
        "n_patients": n_patients,
    }


def save(data):
    """Write the transition dataset to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "transitions.npz")
    np.savez_compressed(
        path,
        states=data["states"],
        doses=data["doses"],
        next_states=data["next_states"],
        patient_ids=data["patient_ids"],
        patient_params=data["patient_params"],
        param_names=np.array(data["param_names"]),
        dt=np.array(data["dt"]),
    )
    return path


def plot_coverage(data):
    """
    Show what regions of the state space the data covers.

    Gaps here become blind spots in the surrogate, so this is worth looking
    at before training rather than after.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    states = data["states"]
    doses = data["doses"]
    deltas = data["next_states"] - states

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    labels = ["N healthy", "T tumour", "I immune", "u drug"]

    # Where in state space the samples sit.
    ax = axes[0, 0]
    ax.hexbin(states[:, 1], states[:, 0], gridsize=45, cmap="magma", bins="log")
    ax.set_xlabel("tumour T"); ax.set_ylabel("healthy N")
    ax.set_title("state space coverage")

    ax = axes[0, 1]
    ax.hexbin(states[:, 1], states[:, 2], gridsize=45, cmap="magma", bins="log")
    ax.set_xlabel("tumour T"); ax.set_ylabel("immune I")
    ax.set_title("tumour vs immune")

    ax = axes[0, 2]
    ax.hist(doses, bins=40, color="#0f4c5c", alpha=0.85)
    ax.set_xlabel("applied dose v"); ax.set_ylabel("count")
    ax.set_title("dose coverage")

    # How much each quantity moves in one step. This is what the surrogate
    # actually predicts, so its scale matters for training.
    for i in range(3):
        ax = axes[1, i]
        ax.hist(deltas[:, i], bins=60, color="#a4243b", alpha=0.85)
        ax.set_xlabel(f"change in {labels[i]} per step")
        ax.set_yscale("log")
        ax.set_title(f"step change, {labels[i]}")

    fig.suptitle(
        f"Surrogate training data: {len(states):,} transitions "
        f"from {data['n_patients']} patients (A2.2)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a2_2_data.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating surrogate training data...")
    print(f"  timestep DT       = {DT}")
    print(f"  horizon           = {T_END}")
    print(f"  schedules/patient = {N_SCHEDULES} randomised + 1 untreated")
    print()

    data = generate()

    states = data["states"]
    deltas = data["next_states"] - states

    print()
    print(f"Transitions generated: {len(states):,}")
    print(f"  from {data['n_patients']} patients")
    print(f"  average {len(states) / data['n_patients']:.0f} transitions per patient")
    print()

    print("State ranges covered:")
    for i, label in enumerate(["N healthy", "T tumour", "I immune", "u drug  "]):
        print(f"  {label}   {states[:, i].min():8.4f} to {states[:, i].max():8.4f}")
    print()

    print("Per-step changes, the quantity the surrogate predicts:")
    for i, label in enumerate(["N healthy", "T tumour", "I immune", "u drug  "]):
        print(f"  {label}   mean abs {np.abs(deltas[:, i]).mean():.5f}   "
              f"max abs {np.abs(deltas[:, i]).max():.5f}")
    print()

    path = save(data)
    fig_path = plot_coverage(data)

    print("Saved:")
    print(f"  {path}")
    print(f"  {fig_path}")
    print()
    print("Data ready. Next step A2.3 splits it by patient, so no patient")
    print("appears in both training and test sets.")
