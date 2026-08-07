"""
Validate the neural surrogate (step A3.3).

Numerical accuracy is not the same as control-relevant accuracy. A surrogate
can have a tiny mean error and still misplace the boundary between immune
control and tumour escape. If it does, a controller trained on it will learn
the wrong stopping point and fail on the real model.

So this script asks four questions, in increasing order of importance:

1. How does rollout error grow over a full horizon, on TREATED trajectories
   as well as untreated ones? Step A3.1 only checked untreated rollouts,
   which are the easy case.

2. Does the surrogate stay physically plausible, or does it drift into
   negative populations and unbounded values?

3. Does a surrogate rollout produce the same OUTCOME as the truth?
   Controlled, escape, or host failure.

4. Does the surrogate reproduce the SEPARATRIX at T0* = 0.155?

Question 4 is the real test of the digital twin. Everything the controller
does depends on the boundary being in the right place.

Outputs
-------
    figures/a3_3_rollout_error.png    error growth over the horizon
    figures/a3_3_outcome_match.png    outcome agreement and separatrix check
    reports/a3_3_validation.md        written summary of the results

Run with:
    python src/validate_surrogate.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Params, default_initial_state
from simulate import simulate, classify
from control import simulate_controlled, constant_dose, V_MAX
from cohort import make_params, PARAMETER_RANGES
from train_surrogate import Surrogate, Normaliser, STATE_LABELS


DATA_DIR = "data"
MODEL_DIR = "models"
FIGURE_DIR = "figures"
REPORT_DIR = "reports"

DT = 0.5
T_END = 100.0
N_STEPS = int(T_END / DT)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

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

    x_norm = Normaliser.from_state_dict(ckpt["x_norm"])
    y_norm = Normaliser.from_state_dict(ckpt["y_norm"])

    return model, x_norm, y_norm


def surrogate_rollout(model, x_norm, y_norm, y0, patient_params, dose_series,
                      device=torch.device("cpu")):
    """
    Roll the surrogate forward using only its own predictions.

    Args:
        y0            : starting state, array of 4.
        patient_params: array of 8 patient parameters.
        dose_series   : array of doses, one per step.

    Returns:
        array of shape (n_steps + 1, 4), the predicted trajectory.
    """
    state = torch.tensor(y0, dtype=torch.float32)
    params_t = torch.tensor(patient_params, dtype=torch.float32)
    trajectory = [state.numpy().copy()]

    with torch.no_grad():
        for dose in dose_series:
            x = torch.cat([
                state,
                torch.tensor([dose], dtype=torch.float32),
                params_t,
            ]).unsqueeze(0)

            delta = y_norm.decode(model(x_norm.encode(x).to(device)).cpu()).squeeze(0)
            state = state + delta
            trajectory.append(state.numpy().copy())

    return np.array(trajectory)


def truth_rollout(y0, params, dose_series):
    """
    Integrate the mechanistic model under the same piecewise-constant doses,
    sampled onto the same DT grid.
    """
    def policy(t, y):
        index = min(int(t // DT), len(dose_series) - 1)
        return float(dose_series[index])

    n_points = N_STEPS * 4 + 1
    t, Y, _v, _total = simulate_controlled(
        y0, policy, params=params, t_end=T_END, n_points=n_points
    )

    stride = max(1, int(round(DT / (t[1] - t[0]))))
    idx = np.arange(0, len(t), stride)[:N_STEPS + 1]
    return Y[:, idx].T


# ----------------------------------------------------------------------
# Question 1 and 2: error growth and physical plausibility
# ----------------------------------------------------------------------

def rollout_comparison(model, x_norm, y_norm, n_patients=40, seed=5):
    """
    Compare surrogate and truth rollouts on unseen test patients, under both
    untreated and treated dosing.
    """
    cohort = np.load(os.path.join(DATA_DIR, "cohort.npz"), allow_pickle=True)
    splits = np.load(os.path.join(DATA_DIR, "splits.npz"), allow_pickle=True)

    param_names = [str(x) for x in cohort["param_names"]]
    test_patients = splits["test_patients"][:n_patients]
    rng = np.random.default_rng(seed)

    records = []

    for pid in test_patients:
        patient_params = cohort["params"][pid]
        y0 = cohort["y0"][pid]
        p = make_params({name: float(patient_params[i])
                         for i, name in enumerate(param_names)})

        schedules = {
            "untreated": np.zeros(N_STEPS),
            "constant": np.full(N_STEPS, 0.4),
            "randomised": _random_dose_series(rng),
        }

        for name, dose_series in schedules.items():
            true_traj = truth_rollout(y0, p, dose_series)
            pred_traj = surrogate_rollout(
                model, x_norm, y_norm, y0, patient_params, dose_series
            )

            n = min(len(true_traj), len(pred_traj))
            error = np.abs(pred_traj[:n] - true_traj[:n])

            records.append({
                "patient": int(pid),
                "schedule": name,
                "true": true_traj[:n],
                "pred": pred_traj[:n],
                "error": error,
                "final_error": error[-1],
                "mean_error": error.mean(axis=0),
                # Physical plausibility of the surrogate's own trajectory.
                "min_value": pred_traj[:n].min(),
                "max_value": pred_traj[:n].max(),
                "true_outcome": classify(true_traj[:n, 1], true_traj[:n, 0]),
                "pred_outcome": classify(pred_traj[:n, 1], pred_traj[:n, 0]),
            })

    return records


def _random_dose_series(rng, hold_steps=10, p_zero=0.3):
    """Piecewise-constant random doses on the DT grid."""
    n_segments = int(np.ceil(N_STEPS / hold_steps))
    doses = rng.uniform(0.0, V_MAX, size=n_segments)
    doses[rng.random(n_segments) < p_zero] = 0.0
    return np.repeat(doses, hold_steps)[:N_STEPS]


# ----------------------------------------------------------------------
# Question 4: does the surrogate reproduce the separatrix?
# ----------------------------------------------------------------------

def surrogate_separatrix(model, x_norm, y_norm, low=0.08, high=0.30,
                         tolerance=1e-3):
    """
    Find the surrogate's separatrix by bisection, using baseline parameters.

    Compared against the mechanistic value of T0* = 0.155 from step A1.2.
    """
    baseline = Params().as_dict()
    param_names = list(PARAMETER_RANGES.keys())
    patient_params = np.array([baseline[name] for name in param_names],
                              dtype=np.float32)

    base_state = default_initial_state()
    zero_doses = np.zeros(N_STEPS)

    def escapes(T0):
        y0 = base_state.copy()
        y0[1] = T0
        traj = surrogate_rollout(model, x_norm, y_norm, y0, patient_params, zero_doses)
        return classify(traj[:, 1], traj[:, 0]) == "escape"

    if escapes(low) or not escapes(high):
        return None, "bisection bracket invalid on the surrogate"

    iterations = 0
    while (high - low) > tolerance and iterations < 60:
        mid = 0.5 * (low + high)
        if escapes(mid):
            high = mid
        else:
            low = mid
        iterations += 1

    return 0.5 * (low + high), None


def truth_separatrix_curve(n=25, low=0.08, high=0.30):
    """Final tumour burden against starting tumour, from the true model."""
    base = default_initial_state()
    p = Params()
    T0_values = np.linspace(low, high, n)

    finals = []
    for T0 in T0_values:
        y0 = base.copy()
        y0[1] = T0
        t, Y = simulate(y0, params=p, v=0.0, t_end=T_END, n_points=400)
        finals.append(Y[1][-1])

    return T0_values, np.array(finals)


def surrogate_separatrix_curve(model, x_norm, y_norm, n=25, low=0.08, high=0.30):
    """The same curve, produced by the surrogate."""
    baseline = Params().as_dict()
    param_names = list(PARAMETER_RANGES.keys())
    patient_params = np.array([baseline[name] for name in param_names],
                              dtype=np.float32)

    base = default_initial_state()
    zero_doses = np.zeros(N_STEPS)
    T0_values = np.linspace(low, high, n)

    finals = []
    for T0 in T0_values:
        y0 = base.copy()
        y0[1] = T0
        traj = surrogate_rollout(model, x_norm, y_norm, y0, patient_params, zero_doses)
        finals.append(traj[-1, 1])

    return T0_values, np.array(finals)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_rollout_error(records):
    """Error growth over the horizon, split by dosing schedule."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    schedules = ["untreated", "constant", "randomised"]
    colours = {"untreated": "#1b7a5a", "constant": "#0f4c5c", "randomised": "#a4243b"}

    # Error growth curves.
    ax = axes[0]
    for schedule in schedules:
        subset = [r for r in records if r["schedule"] == schedule]
        if not subset:
            continue
        stacked = np.stack([r["error"][:, 1] for r in subset])   # tumour error
        steps = np.arange(stacked.shape[1]) * DT
        ax.plot(steps, stacked.mean(axis=0), color=colours[schedule], label=schedule)
        ax.fill_between(steps,
                        np.percentile(stacked, 25, axis=0),
                        np.percentile(stacked, 75, axis=0),
                        color=colours[schedule], alpha=0.2)
    ax.set_xlabel("time"); ax.set_ylabel("absolute error in tumour T")
    ax.set_title("Error growth over the horizon")
    ax.legend(fontsize=8)

    # Final error by dimension and schedule.
    ax = axes[1]
    x = np.arange(len(STATE_LABELS))
    width = 0.26
    for k, schedule in enumerate(schedules):
        subset = [r for r in records if r["schedule"] == schedule]
        if not subset:
            continue
        finals = np.stack([r["final_error"] for r in subset]).mean(axis=0)
        ax.bar(x + (k - 1) * width, finals, width,
               color=colours[schedule], label=schedule, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("mean error at end of horizon")
    ax.set_title("Final error by quantity")
    ax.legend(fontsize=8)

    # A worked example, the worst treated case.
    ax = axes[2]
    treated = [r for r in records if r["schedule"] == "randomised"]
    if treated:
        worst = max(treated, key=lambda r: r["error"][:, 1].max())
        steps = np.arange(len(worst["true"])) * DT
        ax.plot(steps, worst["true"][:, 1], color="#0f4c5c", linewidth=2, label="true T")
        ax.plot(steps, worst["pred"][:, 1], color="#a4243b", linestyle="--",
                linewidth=1.6, label="surrogate T")
        ax.plot(steps, worst["true"][:, 0], color="#1b7a5a", linewidth=2, label="true N")
        ax.plot(steps, worst["pred"][:, 0], color="#c78c3c", linestyle="--",
                linewidth=1.6, label="surrogate N")
        ax.set_xlabel("time"); ax.set_ylabel("population")
        ax.set_title(f"Worst treated case, patient {worst['patient']}")
        ax.legend(fontsize=7)

    fig.suptitle("Surrogate rollout accuracy (A3.3)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a3_3_rollout_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_outcome_and_separatrix(records, truth_curve, surr_curve,
                                truth_sep, surr_sep):
    """Outcome agreement and the separatrix comparison."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Outcome agreement as a confusion-style count.
    ax = axes[0]
    labels = ["controlled", "escape", "host failure", "intermediate"]
    matrix = np.zeros((len(labels), len(labels)))
    for r in records:
        if r["true_outcome"] in labels and r["pred_outcome"] in labels:
            i = labels.index(r["true_outcome"])
            j = labels.index(r["pred_outcome"])
            matrix[i, j] += 1

    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("surrogate outcome"); ax.set_ylabel("true outcome")
    ax.set_title("Outcome agreement")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Separatrix comparison.
    ax = axes[1]
    ax.plot(truth_curve[0], truth_curve[1], "o-", color="#0f4c5c",
            markersize=4, label="mechanistic model")
    ax.plot(surr_curve[0], surr_curve[1], "s--", color="#a4243b",
            markersize=4, label="surrogate")
    if truth_sep is not None:
        ax.axvline(truth_sep, color="#0f4c5c", linestyle=":", linewidth=1.4)
    if surr_sep is not None:
        ax.axvline(surr_sep, color="#a4243b", linestyle=":", linewidth=1.4)
    ax.set_xlabel("starting tumour size T0")
    ax.set_ylabel("final tumour size T")
    title = "Separatrix reproduction"
    if truth_sep is not None and surr_sep is not None:
        title += f"\ntrue {truth_sep:.4f}   surrogate {surr_sep:.4f}"
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.suptitle("Control-relevant validation (A3.3)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a3_3_outcome_match.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    device = torch.device("cpu")
    print("Loading trained surrogate...\n")
    model, x_norm, y_norm = load_surrogate(device)

    print("Rolling out on unseen test patients, three dosing schedules each...")
    records = rollout_comparison(model, x_norm, y_norm, n_patients=40)
    print(f"  {len(records)} rollouts completed\n")

    # Question 1: error over the horizon.
    print("Rollout error over a full horizon (mean over patients):")
    print(f"{'schedule':<14} " + " ".join(f"{l:>12}" for l in STATE_LABELS))
    print("-" * 66)
    for schedule in ["untreated", "constant", "randomised"]:
        subset = [r for r in records if r["schedule"] == schedule]
        if not subset:
            continue
        mean_err = np.stack([r["mean_error"] for r in subset]).mean(axis=0)
        print(f"{schedule:<14} " + " ".join(f"{e:>12.5f}" for e in mean_err))
    print()

    # Question 2: physical plausibility.
    worst_min = min(r["min_value"] for r in records)
    worst_max = max(r["max_value"] for r in records)
    print("Physical plausibility of surrogate trajectories:")
    print(f"  most negative value reached: {worst_min:.5f}")
    print(f"  largest value reached:       {worst_max:.5f}")
    if worst_min < -0.05:
        print("  WARNING: the surrogate drifts meaningfully negative.")
    else:
        print("  No meaningful negative drift.")
    print()

    # Question 3: outcome agreement.
    agree = sum(1 for r in records if r["true_outcome"] == r["pred_outcome"])
    print(f"Outcome agreement: {agree} / {len(records)} "
          f"({100.0 * agree / len(records):.1f} %)")
    disagreements = [r for r in records if r["true_outcome"] != r["pred_outcome"]]
    for r in disagreements[:6]:
        print(f"  patient {r['patient']:>3} ({r['schedule']:<10}) "
              f"true {r['true_outcome']:<13} surrogate {r['pred_outcome']}")
    if len(disagreements) > 6:
        print(f"  ... and {len(disagreements) - 6} more")
    print()

    # Question 4: the separatrix.
    print("Separatrix comparison (the test that matters for control)...")
    truth_sep = 0.15503   # from step A1.2
    surr_sep, err = surrogate_separatrix(model, x_norm, y_norm)

    if surr_sep is None:
        print(f"  could not locate: {err}")
    else:
        diff = abs(surr_sep - truth_sep)
        print(f"  mechanistic model: T0* = {truth_sep:.5f}")
        print(f"  surrogate:         T0* = {surr_sep:.5f}")
        print(f"  difference:              {diff:.5f} "
              f"({100.0 * diff / truth_sep:.2f} % of the true value)")
    print()

    print("Building comparison curves...")
    truth_curve = truth_separatrix_curve()
    surr_curve = surrogate_separatrix_curve(model, x_norm, y_norm)

    p1 = plot_rollout_error(records)
    p2 = plot_outcome_and_separatrix(records, truth_curve, surr_curve,
                                     truth_sep, surr_sep)

    # Written summary.
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "a3_3_validation.md")
    with open(report_path, "w") as f:
        f.write("# Surrogate Validation (A3.3)\n\n")
        f.write(f"Rollouts: {len(records)} across 40 unseen test patients, ")
        f.write("under untreated, constant and randomised dosing.\n\n")
        f.write("## Rollout error over a full horizon\n\n")
        f.write("| schedule | " + " | ".join(STATE_LABELS) + " |\n")
        f.write("|---" * (len(STATE_LABELS) + 1) + "|\n")
        for schedule in ["untreated", "constant", "randomised"]:
            subset = [r for r in records if r["schedule"] == schedule]
            if not subset:
                continue
            mean_err = np.stack([r["mean_error"] for r in subset]).mean(axis=0)
            f.write(f"| {schedule} | " + " | ".join(f"{e:.5f}" for e in mean_err) + " |\n")
        f.write(f"\n## Outcome agreement\n\n{agree} of {len(records)} ")
        f.write(f"({100.0 * agree / len(records):.1f} %)\n\n")
        f.write("## Separatrix\n\n")
        if surr_sep is not None:
            f.write(f"- Mechanistic model: T0* = {truth_sep:.5f}\n")
            f.write(f"- Surrogate: T0* = {surr_sep:.5f}\n")
            f.write(f"- Difference: {abs(surr_sep - truth_sep):.5f}\n")
        else:
            f.write(f"Could not be located: {err}\n")
        f.write("\n## Physical plausibility\n\n")
        f.write(f"- Most negative value: {worst_min:.5f}\n")
        f.write(f"- Largest value: {worst_max:.5f}\n")

    print("\nSaved:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {report_path}")
