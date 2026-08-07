"""
Why are so few patients rescuable? (pre-A4.2 diagnosis)

Step diagnose_env.py established that only about 17.5 percent of patients can
be controlled by any constant dose, that the horizon is not the limitation,
and that the surrogate agrees exactly with the mechanistic model. So the
difficulty is a property of the cohort itself.

Hypothesis
----------
The drug is far too weak to kill the tumour directly. Its kill term is
a2 * (1 - exp(-u)) * T. With a2 around 0.3 and the drug saturating near u = 1,
the maximum kill rate is roughly 0.19 per unit time, against a tumour growth
rate r1 of roughly 1.5.

If that is right, the drug never eliminates the tumour. It only pushes a
patient across the separatrix so the immune system can finish the job. That
works only for patients who HAVE a healthy attractor to be pushed into. For
patients whose immune system is too weak, the system is monostable, the
diseased state is the only stable outcome, and no dose can help.

Three tests
-----------
1. Does each patient have a healthy attractor at all? Tested by starting them
   with a negligible tumour and no treatment. If even that escapes, the
   healthy state does not exist for this patient.

2. Which parameters separate rescuable from unrescuable patients?

3. How potent would the drug need to be? A sweep over the drug efficacy
   parameter a2, showing how the control ceiling responds.

Test 3 matters because a2 was written from memory and flagged as needing
verification against the source paper. This shows how much rests on it.

Run with:
    python src/diagnose_rescuability.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Params
from simulate import simulate, classify
from control import simulate_controlled, V_MAX
from cohort import make_params, NEEDS_RESCUE, PARAMETER_RANGES


DATA_DIR = "data"
FIGURE_DIR = "figures"

N_PATIENTS = 60
T_END = 200.0
TINY_TUMOUR = 0.02      # small enough that any healthy attractor should win


def load_patients(n=N_PATIENTS):
    """Load a sample of patients who need rescue, with their parameters."""
    cohort = np.load(os.path.join(DATA_DIR, "cohort.npz"), allow_pickle=True)
    splits = np.load(os.path.join(DATA_DIR, "splits.npz"), allow_pickle=True)

    outcomes = np.array([str(x) for x in cohort["outcomes"]])
    ids = [i for i in splits["train_patients"] if outcomes[i] in NEEDS_RESCUE][:n]
    param_names = [str(x) for x in cohort["param_names"]]

    patients = []
    for i in ids:
        values = {name: float(cohort["params"][i][k])
                  for k, name in enumerate(param_names)}
        patients.append({
            "id": int(i),
            "values": values,
            "params": make_params(values),
            "y0": cohort["y0"][i].astype(float),
        })

    return patients, param_names


# ----------------------------------------------------------------------
# Test 1: does a healthy attractor exist for this patient?
# ----------------------------------------------------------------------

def has_healthy_attractor(patient):
    """
    Start the patient with a negligible tumour and no treatment.

    If even a tiny tumour escapes, this patient has no healthy stable state,
    the system is monostable, and no dosing strategy can rescue them. The
    drug moves patients between basins; it cannot create a basin.
    """
    y0 = patient["y0"].copy()
    y0[1] = TINY_TUMOUR

    t, Y = simulate(y0, params=patient["params"], v=0.0,
                    t_end=T_END, n_points=800)
    return classify(Y[1], Y[0]) == "controlled"


def personal_separatrix(patient, low=0.005, high=0.60, tolerance=2e-3):
    """
    Locate this patient's own separatrix by bisection.

    Returns None if the patient is monostable, meaning no boundary exists.
    """
    def escapes(T0):
        y0 = patient["y0"].copy()
        y0[1] = T0
        t, Y = simulate(y0, params=patient["params"], v=0.0,
                        t_end=T_END, n_points=500)
        return classify(Y[1], Y[0]) != "controlled"

    if escapes(low):
        return None          # monostable, diseased only
    if not escapes(high):
        return float("inf")  # monostable, healthy only

    iterations = 0
    while (high - low) > tolerance and iterations < 40:
        mid = 0.5 * (low + high)
        if escapes(mid):
            high = mid
        else:
            low = mid
        iterations += 1

    return 0.5 * (low + high)


# ----------------------------------------------------------------------
# Test 3: how potent would the drug need to be?
# ----------------------------------------------------------------------

def control_rate_at_potency(patients, a2_multiplier, dose=0.6):
    """
    Fraction of patients controlled if drug efficacy were scaled by a factor.

    a2 is the drug kill rate on tumour cells. It was written from memory and
    flagged for verification, so its influence on the result matters.
    """
    controlled = 0
    for p in patients:
        scaled = make_params(dict(p["values"]))
        scaled.a2 = p["values"]["a2"] * a2_multiplier

        def policy(t, y):
            return dose

        try:
            t, Y, _v, _d = simulate_controlled(
                p["y0"], policy, params=scaled, t_end=T_END, n_points=800
            )
        except RuntimeError:
            continue

        if classify(Y[1], Y[0]) == "controlled":
            controlled += 1

    return 100.0 * controlled / len(patients)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    patients, param_names = load_patients()
    print(f"Analysing {len(patients)} patients needing rescue.\n")

    # ------------------------------------------------------------------
    print("Test 1. Does each patient have a healthy attractor?")
    print(f"        Starting tumour set to {TINY_TUMOUR}, no treatment.\n")

    for p in patients:
        p["bistable"] = has_healthy_attractor(p)
        p["separatrix"] = personal_separatrix(p) if p["bistable"] else None

    bistable = [p for p in patients if p["bistable"]]
    monostable = [p for p in patients if not p["bistable"]]

    print(f"        Healthy attractor exists:    {len(bistable):>3} / {len(patients)} "
          f"({100.0 * len(bistable) / len(patients):.1f} %)")
    print(f"        Monostable, diseased only:   {len(monostable):>3} / {len(patients)} "
          f"({100.0 * len(monostable) / len(patients):.1f} %)")
    print()
    print("        Patients with no healthy attractor cannot be rescued by any")
    print("        dosing strategy. The drug moves patients between basins; it")
    print("        cannot create a basin that does not exist.")
    print()

    if bistable:
        seps = [p["separatrix"] for p in bistable
                if p["separatrix"] is not None and np.isfinite(p["separatrix"])]
        if seps:
            print(f"        Personal separatrices among rescuable patients:")
            print(f"          range  {min(seps):.4f} to {max(seps):.4f}")
            print(f"          median {np.median(seps):.4f}")
            print(f"        (the baseline patient's was 0.1550)")
            print()
            print("        These vary by patient, which is precisely why a fixed")
            print("        treatment threshold cannot work and state feedback can.")
            print()

    # ------------------------------------------------------------------
    print("Test 2. Which parameters separate the two groups?\n")
    print(f"{'parameter':<10} {'rescuable mean':>16} {'unrescuable mean':>18} {'ratio':>8}")
    print("-" * 56)

    for name in param_names:
        if not bistable or not monostable:
            break
        a = np.mean([p["values"][name] for p in bistable])
        b = np.mean([p["values"][name] for p in monostable])
        ratio = a / b if b != 0 else float("nan")
        flag = "  <-- strong" if abs(ratio - 1.0) > 0.15 else ""
        print(f"{name:<10} {a:>16.4f} {b:>18.4f} {ratio:>8.2f}{flag}")
    print()
    print("        Ratios far from 1.0 identify what makes a patient rescuable.")
    print()

    # ------------------------------------------------------------------
    print("Test 3. How much does drug potency matter?")
    print("        Control rate at v = 0.6 as the drug kill rate a2 is scaled.\n")

    multipliers = [1.0, 2.0, 3.0, 5.0, 8.0]
    potency_results = []

    print(f"{'a2 scaled by':<14} {'control rate':>14}")
    print("-" * 30)
    for m in multipliers:
        rate = control_rate_at_potency(patients, m)
        potency_results.append((m, rate))
        print(f"{m:<14.1f} {rate:>13.1f}%")
    print()

    baseline_a2 = Params().a2
    print(f"        Current a2 = {baseline_a2}. Maximum kill rate is roughly")
    print(f"        a2 * (1 - exp(-1)) = {baseline_a2 * 0.632:.3f} per unit time,")
    print(f"        against a tumour growth rate r1 of {Params().r1}.")
    print()
    print("        a2 was written from memory and flagged for verification. If the")
    print("        source paper gives a substantially different value, this is the")
    print("        parameter to check first, since the whole difficulty rests on it.")
    print()

    # ------------------------------------------------------------------
    # Figure
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar(["healthy attractor\nexists", "monostable\ndiseased"],
           [len(bistable), len(monostable)],
           color=["#1b7a5a", "#4a1c2f"], alpha=0.9)
    ax.set_ylabel("patients")
    ax.set_title("Structural rescuability")

    ax = axes[1]
    if bistable and monostable:
        x = np.arange(len(param_names))
        width = 0.38
        a_vals = [np.mean([p["values"][n] for p in bistable]) for n in param_names]
        b_vals = [np.mean([p["values"][n] for p in monostable]) for n in param_names]
        # Normalise each parameter so they are comparable on one axis.
        scale = [max(a, b) for a, b in zip(a_vals, b_vals)]
        ax.bar(x - width / 2, [a / s for a, s in zip(a_vals, scale)], width,
               label="rescuable", color="#1b7a5a", alpha=0.9)
        ax.bar(x + width / 2, [b / s for b, s in zip(b_vals, scale)], width,
               label="unrescuable", color="#4a1c2f", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("relative mean value")
        ax.set_title("What distinguishes them")
        ax.legend(fontsize=8)

    ax = axes[2]
    ms = [m for m, _ in potency_results]
    rates = [r for _, r in potency_results]
    ax.plot(ms, rates, "o-", color="#a4243b", linewidth=2)
    ax.set_xlabel("drug potency, a2 scaled by")
    ax.set_ylabel("control rate at v = 0.6 (%)")
    ax.set_title("Sensitivity to drug potency")

    fig.suptitle("Why most patients are unrescuable", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a4_diagnosis_rescuability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")
    print()
    print("Read the Test 1 percentage as the true ceiling for any controller.")
    print("If it is close to 17.5 percent, the cohort is structurally hard and the")
    print("honest move is to redefine the task around the rescuable subset. If it")
    print("is much higher, the constant-dose baselines were simply inefficient and")
    print("a learned controller has real room to improve.")
