"""
Integrate the tumour-immune model over time and plot baseline behaviour.

Purpose (step A1.2): model.py only gives the instantaneous rate of change.
This script steps the system forward through time so we can see what
actually happens to a patient, and checks that the model shows the two
regimes it should:

    - small tumours held in check by the immune system  (immune control)
    - large tumours growing away unchecked              (tumour escape)

Finding both regimes confirms the model behaves sensibly, and tells us
which starting tumour size gives a genuine control problem for the rest
of the project.

Run with:
    python src/simulate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # write image files without opening a window
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from model import Params, dynamics, default_initial_state


# Where plots are saved. Path is relative to the project root, so run this
# script from the project folder, not from inside src/.
FIGURE_DIR = "figures"


def simulate(y0, params=None, v=0.0, t_end=100.0, n_points=2000):
    """
    Run the model forward through time.

    Args:
        y0       : starting state, array of [N, T, I, u].
        params   : Params instance. Defaults to the baseline parameters.
        v        : drug injection rate, held constant. Zero means untreated.
        t_end    : how long to simulate for, in model time units.
        n_points : how many time points to record. This only affects the
                   smoothness of the output, not the accuracy, since the
                   solver chooses its own internal step sizes.

    Returns:
        t   : array of time points.
        Y   : array of shape (4, n_points) holding N, T, I, u over time.
    """
    if params is None:
        params = Params()

    # The times we want the answer recorded at.
    t_eval = np.linspace(0.0, t_end, n_points)

    # solve_ivp does the actual work. It repeatedly calls our dynamics
    # function and takes small steps forward, choosing step sizes
    # automatically to keep the answer accurate.
    #
    # "LSODA" is chosen because this system can be stiff, meaning some
    # quantities change much faster than others. LSODA detects that and
    # switches method accordingly, which is more robust than the default.
    solution = solve_ivp(
        fun=lambda t, y: dynamics(t, y, params, v),
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    return solution.t, solution.y


def classify(T_series, N_series, T_low=0.05, T_high=0.40, N_fail=0.20):
    """
    Label the outcome of a trajectory, following the rules in metrics_note.md.

    The thresholds are the provisional ones from step A0.3. Seeing how they
    behave on real trajectories is exactly how we decide whether to keep them.

    Returns one of: "host failure", "controlled", "escape", "intermediate".
    """
    # Host failure overrides everything: if healthy tissue collapses, the
    # outcome is a failure no matter what the tumour did.
    if N_series.min() < N_fail:
        return "host failure"

    T_final = T_series[-1]

    if T_final < T_low:
        return "controlled"
    if T_final > T_high:
        return "escape"
    return "intermediate"


def run_tumour_size_sweep():
    """
    Simulate untreated patients starting from a range of tumour sizes.

    This is the central check of step A1.2. If the model is behaving like
    the published one, small starting tumours should be controlled by the
    immune system and large ones should escape.
    """
    params = Params()
    base_state = default_initial_state()

    # A spread of starting tumour sizes, from very small to substantial.
    starting_tumours = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]

    results = []
    for T0 in starting_tumours:
        y0 = base_state.copy()
        y0[1] = T0  # index 1 is the tumour population

        t, Y = simulate(y0, params=params, v=0.0, t_end=100.0)
        N, T, I, u = Y

        outcome = classify(T, N)
        results.append(
            {
                "T0": T0,
                "t": t,
                "N": N,
                "T": T,
                "I": I,
                "outcome": outcome,
                "T_final": T[-1],
                "N_min": N.min(),
            }
        )

    return results


def plot_sweep(results):
    """Plot tumour trajectories for each starting size, on one set of axes."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: tumour population over time.
    ax = axes[0]
    for r in results:
        ax.plot(r["t"], r["T"], label=f"T0 = {r['T0']:.2f}  ({r['outcome']})")
    ax.axhline(0.05, linestyle=":", linewidth=1, color="grey")
    ax.axhline(0.40, linestyle=":", linewidth=1, color="grey")
    ax.set_xlabel("time")
    ax.set_ylabel("tumour population T")
    ax.set_title("Untreated tumour trajectories")
    ax.legend(fontsize=8)

    # Right panel: healthy tissue over time, so we can see the cost.
    ax = axes[1]
    for r in results:
        ax.plot(r["t"], r["N"], label=f"T0 = {r['T0']:.2f}")
    ax.axhline(0.20, linestyle=":", linewidth=1, color="grey")
    ax.set_xlabel("time")
    ax.set_ylabel("healthy cell population N")
    ax.set_title("Healthy tissue, untreated")
    ax.legend(fontsize=8)

    fig.suptitle("Baseline behaviour with no treatment (A1.2)")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a1_2_untreated_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_single_trajectory(T0=0.25):
    """
    Plot all four quantities for one untreated patient.

    Useful for seeing how the populations interact, rather than just
    watching the tumour.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    y0 = default_initial_state()
    y0[1] = T0
    t, Y = simulate(y0, v=0.0, t_end=100.0)
    N, T, I, u = Y

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, N, label="N, healthy cells")
    ax.plot(t, T, label="T, tumour cells")
    ax.plot(t, I, label="I, immune cells")
    ax.set_xlabel("time")
    ax.set_ylabel("population (normalised)")
    ax.set_title(f"Untreated patient, starting tumour T0 = {T0:.2f}")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, f"a1_2_single_T0_{T0:.2f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Running untreated tumour size sweep...\n")

    results = run_tumour_size_sweep()

    # Print a summary table.
    print(f"{'T0':>6}  {'final T':>9}  {'min N':>8}  outcome")
    print("-" * 44)
    for r in results:
        print(
            f"{r['T0']:>6.2f}  {r['T_final']:>9.4f}  {r['N_min']:>8.4f}  {r['outcome']}"
        )
    print()

    sweep_path = plot_sweep(results)
    single_path = plot_single_trajectory(T0=0.25)

    print(f"Saved: {sweep_path}")
    print(f"Saved: {single_path}")
    print()

    # Report whether both regimes appeared, which is the point of this step.
    outcomes = {r["outcome"] for r in results}
    if "escape" in outcomes and "controlled" in outcomes:
        print("Both regimes found: immune control at small tumours, escape at large.")
        print("The model is behaving as expected. Pick a starting tumour size that")
        print("escapes, and use it as the patient the controller must rescue.")
    else:
        print("Only these outcomes appeared:", ", ".join(sorted(outcomes)))
        print("We may need to widen the range of starting tumour sizes, or revisit")
        print("the parameters, before moving on to A2.")
