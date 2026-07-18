"""
Control interface for the tumour-immune model.

Purpose (step A1.3): make the drug input something that can be driven
externally, rather than a fixed number.

A treatment is a POLICY: a function

    v(t, y) -> dose

that receives the current time t and the current state y = [N, T, I, u],
and returns the drug injection rate to apply at that moment.

This signature is chosen deliberately. A constant dose, a pulsed schedule,
a rule of thumb, and a trained reinforcement learning agent are all just
different functions with this same shape. When the learned controller
arrives in Phase 4, it drops straight into this interface.

Run with:
    python src/control.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from model import Params, dynamics, default_initial_state
from simulate import classify

# numpy renamed trapz to trapezoid in version 2.0. This works either way.
_integrate = getattr(np, "trapezoid", None) or np.trapz

FIGURE_DIR = "figures"

# Upper limit on the injection rate. No policy may exceed this. A real
# treatment cannot deliver unlimited drug, and without a cap the control
# problem becomes trivial.
V_MAX = 1.0


# ----------------------------------------------------------------------
# Policies: each is a function v(t, y) -> dose
# ----------------------------------------------------------------------

def no_treatment(t, y):
    """Give nothing. The untreated baseline."""
    return 0.0


def constant_dose(v):
    """
    Give the same dose at all times.

    The simplest non-trivial strategy, and the main baseline the learned
    controller must beat on cumulative drug used.
    """
    def policy(t, y):
        return v
    return policy


def pulsed_dose(v, on_duration, off_duration):
    """
    Alternate between treating and resting.

    Rest periods let healthy tissue and immune cells recover, which is why
    real chemotherapy is given in cycles rather than continuously.
    """
    period = on_duration + off_duration

    def policy(t, y):
        phase = t % period
        return v if phase < on_duration else 0.0
    return policy


def treat_until_threshold(v, T_stop):
    """
    Treat while the tumour is above T_stop, then stop completely.

    This is the policy the bistability result suggests. The controller does
    not need to eliminate the tumour, only push it across the separatrix at
    T0* = 0.155. After that the immune system finishes the job unaided.

    This policy reads the state y, so it is a feedback controller rather
    than a fixed schedule.
    """
    def policy(t, y):
        T = y[1]
        return v if T > T_stop else 0.0
    return policy


# ----------------------------------------------------------------------
# Simulation with a policy
# ----------------------------------------------------------------------

def simulate_controlled(y0, policy, params=None, t_end=100.0, n_points=2000):
    """
    Run the model forward while applying a treatment policy.

    Returns:
        t          : time points.
        Y          : array (4, n_points) of N, T, I, u over time.
        v_series   : the dose applied at each recorded time point.
        total_dose : cumulative drug delivered, the integral of v over time.
    """
    if params is None:
        params = Params()

    def rhs(t, y):
        # Ask the policy for a dose, then clip it to the allowed range so
        # no policy can exceed the dose limit, including a learned one.
        v = float(np.clip(float(policy(t, y)), 0.0, V_MAX))
        return dynamics(t, y, params, v)

    t_eval = np.linspace(0.0, t_end, n_points)

    solution = solve_ivp(
        fun=rhs,
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    t, Y = solution.t, solution.y

    v_series = np.array(
        [float(np.clip(policy(ti, Y[:, i]), 0.0, V_MAX)) for i, ti in enumerate(t)]
    )
    total_dose = float(_integrate(v_series, t))

    return t, Y, v_series, total_dose


def evaluate(y0, policy, label, params=None, t_end=100.0):
    """Run a policy and summarise the outcome using the A0.3 metrics."""
    t, Y, v_series, total_dose = simulate_controlled(
        y0, policy, params=params, t_end=t_end
    )
    N, T, I, u = Y
    return {
        "label": label,
        "t": t, "N": N, "T": T, "I": I, "u": u, "v": v_series,
        "total_dose": total_dose,
        "outcome": classify(T, N),
        "T_final": float(T[-1]),
        "N_min": float(N.min()),
    }


# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------

def dose_sweep(T0=0.25):
    """Find the smallest constant dose that rescues an escaping patient."""
    y0 = default_initial_state()
    y0[1] = T0
    doses = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    return [evaluate(y0, constant_dose(v), f"constant v = {v:.1f}") for v in doses]


def policy_comparison(T0=0.25):
    """
    Compare treatment strategies on the same patient.

    The interesting comparison is between a constant dose held for the whole
    horizon, and a feedback policy that stops once the tumour has crossed
    the separatrix.
    """
    y0 = default_initial_state()
    y0[1] = T0
    return [
        evaluate(y0, no_treatment, "no treatment"),
        evaluate(y0, constant_dose(0.5), "constant, v = 0.5"),
        evaluate(y0, pulsed_dose(0.7, 5.0, 5.0), "pulsed, v = 0.7"),
        evaluate(y0, treat_until_threshold(0.5, 0.10), "treat until T < 0.10, then stop"),
    ]


def plot_comparison(results, filename, title):
    """Plot tumour, healthy tissue, and applied dose for each policy."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for r in results:
        ax.plot(r["t"], r["T"], label=f"{r['label']} ({r['outcome']})")
    ax.axhline(0.05, linestyle=":", linewidth=1, color="grey")
    ax.axhline(0.40, linestyle=":", linewidth=1, color="grey")
    ax.set_xlabel("time"); ax.set_ylabel("tumour population T")
    ax.set_title("Tumour"); ax.legend(fontsize=7)

    ax = axes[1]
    for r in results:
        ax.plot(r["t"], r["N"], label=r["label"])
    ax.axhline(0.20, linestyle=":", linewidth=1, color="grey")
    ax.set_xlabel("time"); ax.set_ylabel("healthy cell population N")
    ax.set_title("Healthy tissue"); ax.legend(fontsize=7)

    ax = axes[2]
    for r in results:
        ax.plot(r["t"], r["v"], label=f"{r['label']} (total {r['total_dose']:.1f})")
    ax.set_xlabel("time"); ax.set_ylabel("injection rate v")
    ax.set_title("Applied dose"); ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_table(results, heading):
    """Print a summary table of policy outcomes."""
    print(heading)
    print(f"{'policy':<34}  {'final T':>8}  {'min N':>7}  {'dose':>7}  outcome")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:<34}  {r['T_final']:>8.4f}  {r['N_min']:>7.4f}  "
              f"{r['total_dose']:>7.2f}  {r['outcome']}")
    print()


if __name__ == "__main__":
    T0 = 0.25  # a patient who escapes if left untreated
    print(f"Escaping patient, starting tumour T0 = {T0}\n")

    sweep = dose_sweep(T0)
    print_table(sweep, "Constant dose sweep")

    rescued = [r for r in sweep if r["outcome"] == "controlled"]
    if rescued:
        cheapest = min(rescued, key=lambda r: r["total_dose"])
        print(f"Smallest constant dose that rescues this patient: {cheapest['label']}")
        print(f"  cumulative drug used: {cheapest['total_dose']:.2f}")
    else:
        print("No constant dose in the tested range rescued this patient.")
    print()

    comparison = policy_comparison(T0)
    print_table(comparison, "Policy comparison")

    p1 = plot_comparison(sweep, "a1_3_dose_sweep.png", f"Constant dose sweep, T0 = {T0} (A1.3)")
    p2 = plot_comparison(comparison, "a1_3_policy_comparison.png", f"Policy comparison, T0 = {T0} (A1.3)")

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print()
    print("Control interface ready. Any function v(t, y) -> dose can now drive")
    print("the model, including the learned controller in Phase 4.")
