"""
Generate a cohort of virtual patients (step A2.1).

A surrogate trained on a single patient learns a single trajectory. To learn
generalisable dynamics we need a population: patients whose tumours grow at
different rates, whose immune systems are stronger or weaker, and who respond
to the drug differently.

Method
------
Parameters are drawn by Latin hypercube sampling. Purely random sampling
leaves clumps and gaps by chance. Latin hypercube divides each parameter
range into equal slices and uses every slice exactly once, giving even
coverage from far fewer samples.

Design decision: patients are LABELLED, not filtered
----------------------------------------------------
An earlier version kept only patients whose tumour escaped untreated. That
was wrong for two reasons.

First, it discarded the sickest patients. Screening flagged around a fifth
of candidates as "host failure" with no treatment at all, meaning the
disease itself destroys healthy tissue. Those are the patients most in need
of rescue, and excluding them made the cohort easier than reality.

Second, the surrogate and the controller need different populations. The
surrogate learns dynamics and should see the whole state space, including
patients the immune system handles alone. Training it only on escapers
biases it against exactly the region a successful treatment ends up in.
The controller does want only patients needing rescue.

So every sampled patient is kept, tagged with its untreated outcome, and
each later phase selects the subset it needs.

Outputs
-------
    configs/cohort_config.json  sampling specification, committed to git
    data/cohort.npz             the cohort, regenerable so not committed
    figures/a2_1_cohort.png     diversity of the population

Run with:
    python src/cohort.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

from model import Params, default_initial_state
from simulate import simulate, classify


CONFIG_DIR = "configs"
DATA_DIR = "data"
FIGURE_DIR = "figures"

RANDOM_SEED = 42

# Outcomes that mean the patient needs treatment. Used by Phase 4 to select
# the controller's training population.
NEEDS_RESCUE = ("escape", "host failure")


# Which parameters vary between patients, as (low, high) multipliers of the
# baseline value. Carrying capacities b1 and b2 are held fixed because they
# define the model's normalisation rather than a property of the patient.
PARAMETER_RANGES = {
    "r1":  (0.70, 1.30),   # tumour growth rate, how aggressive the cancer is
    "c2":  (0.60, 1.40),   # immune kill rate, how effective the immune attack is
    "c1":  (0.75, 1.25),   # immune exhaustion, how fast the tumour wears it down
    "s":   (0.70, 1.30),   # baseline immune supply, constitutional strength
    "rho": (0.50, 1.50),   # immune recruitment in response to the tumour
    "d1":  (0.80, 1.20),   # immune cell death rate
    "a2":  (0.70, 1.30),   # drug effect on tumour, patient drug sensitivity
    "a3":  (0.70, 1.30),   # drug damage to healthy tissue, patient toxicity
}

# Initial conditions, as absolute ranges rather than multipliers.
INITIAL_RANGES = {
    "T0": (0.08, 0.45),    # tumour size at presentation, spanning the separatrix
    "I0": (0.10, 0.25),    # immune level at presentation
}


def build_config(n_requested):
    """Assemble the sampling specification as a plain dictionary."""
    return {
        "random_seed": RANDOM_SEED,
        "n_requested": n_requested,
        "sampling_method": "Latin hypercube",
        "parameter_ranges_as_baseline_multipliers": PARAMETER_RANGES,
        "initial_condition_ranges": INITIAL_RANGES,
        "baseline_parameters": Params().as_dict(),
        "policy": "all patients kept and labelled by untreated outcome",
        "needs_rescue_outcomes": list(NEEDS_RESCUE),
        "horizon": 100.0,
    }


def sample_patients(n_requested, seed=RANDOM_SEED):
    """Draw candidate patients by Latin hypercube sampling."""
    param_names = list(PARAMETER_RANGES.keys())
    init_names = list(INITIAL_RANGES.keys())
    n_dimensions = len(param_names) + len(init_names)

    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    unit_samples = sampler.random(n=n_requested)

    lows = [PARAMETER_RANGES[k][0] for k in param_names] + \
           [INITIAL_RANGES[k][0] for k in init_names]
    highs = [PARAMETER_RANGES[k][1] for k in param_names] + \
            [INITIAL_RANGES[k][1] for k in init_names]

    scaled = qmc.scale(unit_samples, lows, highs)

    baseline = Params().as_dict()
    base_state = default_initial_state()

    param_dicts, y0_rows = [], []
    for row in scaled:
        patient_params = {
            name: baseline[name] * row[i] for i, name in enumerate(param_names)
        }
        param_dicts.append(patient_params)

        offset = len(param_names)
        y0 = base_state.copy()
        y0[1] = row[offset + init_names.index("T0")]   # tumour
        y0[2] = row[offset + init_names.index("I0")]   # immune
        y0_rows.append(y0)

    return param_dicts, np.array(y0_rows)


def make_params(patient_params):
    """Build a Params object with this patient's values substituted in."""
    p = Params()
    for name, value in patient_params.items():
        setattr(p, name, value)
    return p


def generate_cohort(n_requested=400, t_end=100.0, seed=RANDOM_SEED, verbose=True):
    """
    Sample patients, simulate each untreated, and label the outcome.

    Every patient is kept. The untreated outcome is recorded so later phases
    can select the subset they need.
    """
    param_dicts, y0_array = sample_patients(n_requested, seed=seed)
    param_names = list(PARAMETER_RANGES.keys())

    params_rows, outcomes, final_T, min_N = [], [], [], []
    counts = {}

    for i, (patient_params, y0) in enumerate(zip(param_dicts, y0_array)):
        p = make_params(patient_params)
        t, Y = simulate(y0, params=p, v=0.0, t_end=t_end, n_points=400)
        N, T = Y[0], Y[1]

        outcome = classify(T, N)
        counts[outcome] = counts.get(outcome, 0) + 1

        params_rows.append([patient_params[k] for k in param_names])
        outcomes.append(outcome)
        final_T.append(float(T[-1]))
        min_N.append(float(N.min()))

        if verbose and (i + 1) % 100 == 0:
            print(f"  simulated {i + 1} / {n_requested} patients...")

    outcomes = np.array(outcomes)
    needs_rescue = np.isin(outcomes, NEEDS_RESCUE)

    return {
        "param_names": param_names,
        "params": np.array(params_rows),
        "y0": y0_array,
        "outcomes": outcomes,
        "needs_rescue": needs_rescue,
        "final_T": np.array(final_T),
        "min_N": np.array(min_N),
        "n_total": n_requested,
        "outcome_counts": counts,
    }


def save_cohort(cohort, config):
    """Write the cohort data and the sampling configuration to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    data_path = os.path.join(DATA_DIR, "cohort.npz")
    np.savez_compressed(
        data_path,
        param_names=np.array(cohort["param_names"]),
        params=cohort["params"],
        y0=cohort["y0"],
        outcomes=cohort["outcomes"],
        needs_rescue=cohort["needs_rescue"],
        final_T=cohort["final_T"],
        min_N=cohort["min_N"],
    )

    config = dict(config)
    config["outcome_counts"] = cohort["outcome_counts"]
    config["n_needs_rescue"] = int(cohort["needs_rescue"].sum())

    config_path = os.path.join(CONFIG_DIR, "cohort_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return data_path, config_path


def plot_cohort(cohort):
    """Show the diversity of the population and how outcomes are distributed."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    params = cohort["params"]
    names = cohort["param_names"]
    outcomes = cohort["outcomes"]

    colours = {
        "controlled": "#1b7a5a",
        "escape": "#a4243b",
        "host failure": "#4a1c2f",
        "intermediate": "#c78c3c",
    }

    fig = plt.figure(figsize=(14, 8))

    # Parameter distributions, confirming even coverage of each range.
    for i, name in enumerate(names):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.hist(params[:, i], bins=20, color="#0f4c5c", alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)

    # Starting tumour, coloured by untreated outcome.
    ax = fig.add_subplot(3, 4, 9)
    for outcome, colour in colours.items():
        mask = outcomes == outcome
        if mask.any():
            ax.hist(cohort["y0"][mask, 1], bins=18, color=colour, alpha=0.75, label=outcome)
    ax.set_title("starting tumour T0", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)

    ax = fig.add_subplot(3, 4, 10)
    ax.hist(cohort["y0"][:, 2], bins=20, color="#a4243b", alpha=0.85)
    ax.set_title("starting immune I0", fontsize=10)
    ax.tick_params(labelsize=7)

    # Outcome counts.
    ax = fig.add_subplot(3, 4, 11)
    labels = list(cohort["outcome_counts"].keys())
    values = [cohort["outcome_counts"][k] for k in labels]
    ax.bar(range(len(labels)), values,
           color=[colours.get(k, "grey") for k in labels], alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    ax.set_title("untreated outcomes", fontsize=10)
    ax.tick_params(labelsize=7)

    # Where each patient sits relative to the separatrix.
    ax = fig.add_subplot(3, 4, 12)
    for outcome, colour in colours.items():
        mask = outcomes == outcome
        if mask.any():
            ax.scatter(cohort["y0"][mask, 1], cohort["min_N"][mask],
                       s=8, alpha=0.6, color=colour)
    ax.axvline(0.1550, linestyle="--", linewidth=1, color="#0f4c5c")
    ax.axhline(0.20, linestyle=":", linewidth=1, color="grey")
    ax.set_xlabel("starting T0", fontsize=8)
    ax.set_ylabel("untreated min N", fontsize=8)
    ax.set_title("severity vs separatrix", fontsize=10)
    ax.tick_params(labelsize=7)

    n_rescue = int(cohort["needs_rescue"].sum())
    fig.suptitle(
        f"Virtual patient cohort: {cohort['n_total']} patients, "
        f"{n_rescue} needing rescue (A2.1)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a2_1_cohort.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    N_REQUESTED = 400

    print(f"Sampling {N_REQUESTED} patients by Latin hypercube...\n")
    cohort = generate_cohort(n_requested=N_REQUESTED)

    print()
    print("Untreated outcomes across the cohort:")
    for outcome, count in sorted(cohort["outcome_counts"].items()):
        share = 100.0 * count / cohort["n_total"]
        marker = "  <- needs rescue" if outcome in NEEDS_RESCUE else ""
        print(f"  {outcome:<14} {count:>4}   ({share:5.1f} %){marker}")
    print()

    n_rescue = int(cohort["needs_rescue"].sum())
    print(f"Total kept for surrogate training (Phase 3): {cohort['n_total']}")
    print(f"Subset needing rescue, for the controller (Phase 4): {n_rescue} "
          f"({100.0 * n_rescue / cohort['n_total']:.1f} %)")
    print()

    print("Diversity of the full cohort:")
    print(f"  starting tumour T0    {cohort['y0'][:, 1].min():.3f} to {cohort['y0'][:, 1].max():.3f}")
    print(f"  starting immune I0    {cohort['y0'][:, 2].min():.3f} to {cohort['y0'][:, 2].max():.3f}")
    print(f"  untreated final T     {cohort['final_T'].min():.3f} to {cohort['final_T'].max():.3f}")
    print(f"  untreated min N       {cohort['min_N'].min():.3f} to {cohort['min_N'].max():.3f}")
    print()

    config = build_config(N_REQUESTED)
    data_path, config_path = save_cohort(cohort, config)
    fig_path = plot_cohort(cohort)

    print("Saved:")
    print(f"  {data_path}")
    print(f"  {config_path}")
    print(f"  {fig_path}")
    print()
    print("Cohort ready. Next step A2.2 simulates each patient under randomised")
    print("dosing schedules to produce the surrogate's training data.")
