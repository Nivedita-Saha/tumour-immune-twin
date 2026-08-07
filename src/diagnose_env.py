"""
Diagnose the control environment before training (pre-A4.2).

The A4.1 baselines gave a maximum control rate of 11.8 percent, even under
sustained maximum dose. That is low enough to suspect the environment rather
than the policies. Training a controller inside a misconfigured environment
would waste days and produce a result that means nothing.

Three hypotheses are tested here, all against the mechanistic model rather
than the surrogate, so the surrogate cannot mask a problem.

H1. The horizon is too short.
    Step A1.2 showed patients just below the separatrix take 60 to 80 time
    units to collapse. A treated patient must first be pushed across the
    boundary, and only then begins that slow collapse. If the episode ends
    at t = 100, patients on a good trajectory may be scored as failures
    before they finish. Tested by rerunning with a much longer horizon.

H2. The surrogate is extrapolating.
    Training data used doses redrawn every 5 time units with 30 percent
    zeros, and never contained 200 consecutive steps at maximum dose.
    Tested by comparing surrogate and mechanistic control rates on the
    same patients under the same policies.

H3. The problem is genuinely hard.
    If neither of the above explains it, the cohort simply contains many
    unrescuable patients. Tested by measuring how many patients are still
    improving at the end of the horizon, and how many are beyond help.

Run with:
    python src/diagnose_env.py
"""

import os
import numpy as np
import torch

from model import Params
from control import simulate_controlled, V_MAX
from cohort import make_params, NEEDS_RESCUE
from env import TumourImmuneEnv, run_episode, N_FAIL, T_CONTROLLED


DATA_DIR = "data"

N_PATIENTS = 40          # enough to be indicative, small enough to be quick
HORIZON_SHORT = 100.0    # the current episode length
HORIZON_LONG = 400.0     # long enough for slow collapses to finish


def load_rescue_patients(split="train", n=N_PATIENTS):
    """Load patients from one split who need rescue."""
    cohort = np.load(os.path.join(DATA_DIR, "cohort.npz"), allow_pickle=True)
    splits = np.load(os.path.join(DATA_DIR, "splits.npz"), allow_pickle=True)

    outcomes = np.array([str(x) for x in cohort["outcomes"]])
    ids = [i for i in splits[f"{split}_patients"] if outcomes[i] in NEEDS_RESCUE][:n]

    param_names = [str(x) for x in cohort["param_names"]]
    return [
        {
            "id": int(i),
            "params": make_params({
                name: float(cohort["params"][i][k])
                for k, name in enumerate(param_names)
            }),
            "y0": cohort["y0"][i].astype(float),
            "untreated_outcome": outcomes[i],
        }
        for i in ids
    ]


def truth_episode(patient, dose, t_end):
    """
    Run one patient under a constant dose on the mechanistic model.

    Returns the outcome plus diagnostic information about whether the patient
    was still improving when the horizon ended.
    """
    def policy(t, y):
        return dose

    n_points = int(t_end * 8) + 1
    t, Y, v_series, total_dose = simulate_controlled(
        patient["y0"], policy, params=patient["params"],
        t_end=t_end, n_points=n_points,
    )

    N, T = Y[0], Y[1]
    min_N = float(N.min())
    final_T = float(T[-1])

    if min_N < N_FAIL:
        outcome = "host failure"
    elif final_T < T_CONTROLLED:
        outcome = "controlled"
    else:
        outcome = "escape"

    # Is the tumour still falling at the end? A patient whose tumour is
    # decreasing has not failed, it has simply not finished.
    tail = max(1, len(T) // 20)
    slope = float(T[-1] - T[-tail])
    still_improving = slope < -1e-4

    return {
        "outcome": outcome,
        "final_T": final_T,
        "min_N": min_N,
        "total_dose": float(total_dose),
        "still_improving": still_improving,
        "peak_T": float(T.max()),
    }


def summarise(results, label):
    """Aggregate a list of episode results."""
    n = len(results)
    controlled = sum(1 for r in results if r["outcome"] == "controlled")
    failures = sum(1 for r in results if r["outcome"] == "host failure")
    improving = sum(1 for r in results
                    if r["outcome"] == "escape" and r["still_improving"])
    return {
        "label": label,
        "n": n,
        "control_rate": 100.0 * controlled / n,
        "failure_rate": 100.0 * failures / n,
        "improving_rate": 100.0 * improving / n,
        "mean_dose": float(np.mean([r["total_dose"] for r in results])),
        "mean_final_T": float(np.mean([r["final_T"] for r in results])),
    }


def print_table(rows, heading):
    print(heading)
    print(f"{'condition':<34} {'controlled':>11} {'host fail':>10} "
          f"{'improving':>10} {'mean T':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['label']:<34} {r['control_rate']:>10.1f}% {r['failure_rate']:>9.1f}% "
              f"{r['improving_rate']:>9.1f}% {r['mean_final_T']:>8.4f}")
    print()


if __name__ == "__main__":
    patients = load_rescue_patients(n=N_PATIENTS)
    print(f"Diagnosing on {len(patients)} training patients needing rescue.\n")

    doses = [0.0, 0.3, 0.5, V_MAX]

    # ------------------------------------------------------------------
    # H1: is the horizon too short?
    # ------------------------------------------------------------------
    print("H1. Testing whether the horizon is too short.")
    print("    Mechanistic model, same patients, two horizons.\n")

    short_rows, long_rows = [], []
    for dose in doses:
        short = [truth_episode(p, dose, HORIZON_SHORT) for p in patients]
        long = [truth_episode(p, dose, HORIZON_LONG) for p in patients]
        short_rows.append(summarise(short, f"truth, v={dose:.1f}, t_end={HORIZON_SHORT:.0f}"))
        long_rows.append(summarise(long, f"truth, v={dose:.1f}, t_end={HORIZON_LONG:.0f}"))

    print_table(short_rows, f"Horizon {HORIZON_SHORT:.0f} (current episode length)")
    print_table(long_rows, f"Horizon {HORIZON_LONG:.0f} (extended)")

    best_short = max(r["control_rate"] for r in short_rows)
    best_long = max(r["control_rate"] for r in long_rows)
    print(f"    Best control rate at t_end={HORIZON_SHORT:.0f}: {best_short:.1f} %")
    print(f"    Best control rate at t_end={HORIZON_LONG:.0f}: {best_long:.1f} %")
    if best_long > best_short + 15.0:
        print("    VERDICT: the horizon is too short. Many patients were on a")
        print("    successful trajectory but had not finished. Lengthen the episode.")
    else:
        print("    VERDICT: lengthening the horizon does not rescue many patients.")
    print()

    # ------------------------------------------------------------------
    # H2: is the surrogate extrapolating?
    # ------------------------------------------------------------------
    print("H2. Testing whether the surrogate disagrees with the mechanistic model.")
    print("    Same patients, same constant doses, surrogate vs truth.\n")

    env = TumourImmuneEnv(split="train", include_params=True, seed=0)
    # Map cohort patient ids to indices within the environment's pool.
    id_to_index = {int(pid): k for k, pid in enumerate(env.patient_ids)}

    print(f"{'dose':<8} {'surrogate controlled':>21} {'truth controlled':>18} {'gap':>8}")
    print("-" * 58)
    for dose in doses:
        surr_results = []
        for p in patients:
            idx = id_to_index.get(p["id"])
            if idx is None:
                continue
            surr_results.append(
                run_episode(env, lambda obs, d=dose: d, patient_index=idx)
            )
        if not surr_results:
            continue

        surr_rate = 100.0 * sum(1 for r in surr_results
                                if r["outcome"] == "controlled") / len(surr_results)
        truth_rate = summarise(
            [truth_episode(p, dose, HORIZON_SHORT) for p in patients],
            "truth",
        )["control_rate"]

        print(f"v={dose:<5.1f} {surr_rate:>20.1f}% {truth_rate:>17.1f}% "
              f"{surr_rate - truth_rate:>+7.1f}")
    print()
    print("    A gap of more than a few points means the surrogate is being")
    print("    driven outside the region its training data covered.")
    print()

    # ------------------------------------------------------------------
    # H3: how hard is the problem really?
    # ------------------------------------------------------------------
    print("H3. Assessing intrinsic difficulty.")
    print("    For each patient, the best outcome achievable by any constant dose,")
    print("    on the mechanistic model with the extended horizon.\n")

    best_per_patient = []
    for p in patients:
        attempts = [truth_episode(p, d, HORIZON_LONG) for d in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]]
        controlled = [a for a in attempts if a["outcome"] == "controlled"]
        if controlled:
            best = min(controlled, key=lambda a: a["total_dose"])
            best_per_patient.append(("rescuable", best))
        else:
            best = min(attempts, key=lambda a: a["final_T"])
            best_per_patient.append(("not rescuable", best))

    rescuable = [b for label, b in best_per_patient if label == "rescuable"]
    n_rescuable = len(rescuable)

    print(f"    Rescuable by some constant dose: {n_rescuable} / {len(patients)} "
          f"({100.0 * n_rescuable / len(patients):.1f} %)")
    if rescuable:
        print(f"    Of those, mean minimum dose needed: "
              f"{np.mean([b['total_dose'] for b in rescuable]):.2f}")
    print()
    print("    This is the practical ceiling for a constant-dose policy. A learned")
    print("    controller should match or exceed it, and should need less drug,")
    print("    because it can vary the dose over time and across patients.")
    print()

    # ------------------------------------------------------------------
    print("Summary of what to change before training:")
    if best_long > best_short + 15.0:
        print(f"  - Lengthen the episode. t_end = {HORIZON_LONG:.0f} rather than "
              f"{HORIZON_SHORT:.0f}.")
        print("    This requires regenerating training data over the longer horizon,")
        print("    since the surrogate has only seen trajectories up to t = 100.")
    print("  - Reconsider the 80 percent control target in metrics_note.md against")
    print("    the measured ceiling above. Adjust the target with justification,")
    print("    rather than adjusting it later to match whatever the agent achieves.")
