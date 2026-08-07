"""
Aggregate the sweep results into a frontier summary.

Reads every hpc/results/task_*.json, groups them by w_dose, and averages
control_rate and mean_dose across the five seeds. Because each operating
point is now an average over seeds rather than a single run, the frontier
comes with a standard deviation: a more honest picture than one seed.

Writes:
    hpc/reports/frontier_aggregated.json   the numbers
    hpc/reports/frontier_aggregated.png    the plot

Run with:
    python hpc/aggregate.py
"""

import os
import json
import glob
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
REPORTS_DIR = os.path.join(HERE, "reports")


def load_results():
    """Read every task result file into a list of dicts."""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "task_*.json")))
    if not paths:
        raise SystemExit(
            "No result files found in hpc/results/. "
            "Run the sweep first (python hpc/run_one.py --task N)."
        )
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results, paths


def group_by_w_dose(results):
    """Collect control_rate and mean_dose lists keyed by w_dose."""
    grouped = defaultdict(lambda: {"control_rate": [], "mean_dose": [], "seeds": []})
    for r in results:
        w = r["w_dose"]
        grouped[w]["control_rate"].append(r["control_rate"])
        grouped[w]["mean_dose"].append(r["mean_dose"])
        grouped[w]["seeds"].append(r["seed"])
    return grouped


def summarise(grouped):
    """Average across seeds for each w_dose, with standard deviations."""
    rows = []
    for w in sorted(grouped.keys()):
        g = grouped[w]
        rows.append({
            "w_dose": w,
            "n_seeds": len(g["seeds"]),
            "control_rate_mean": float(np.mean(g["control_rate"])),
            "control_rate_std": float(np.std(g["control_rate"])),
            "mean_dose_mean": float(np.mean(g["mean_dose"])),
            "mean_dose_std": float(np.std(g["mean_dose"])),
        })
    return rows


def plot_frontier(rows, path):
    """Plot averaged control rate against averaged dose, with error bars."""
    doses = [r["mean_dose_mean"] for r in rows]
    dose_err = [r["mean_dose_std"] for r in rows]
    rates = [r["control_rate_mean"] for r in rows]
    rate_err = [r["control_rate_std"] for r in rows]

    order = np.argsort(doses)
    doses = np.array(doses)[order]
    dose_err = np.array(dose_err)[order]
    rates = np.array(rates)[order]
    rate_err = np.array(rate_err)[order]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(doses, rates, xerr=dose_err, yerr=rate_err,
                fmt="*-", color="#a4243b", linewidth=2.0, markersize=15,
                markeredgecolor="white", markeredgewidth=1.0,
                ecolor="#a4243b", elinewidth=1.2, capsize=4,
                label="learned controllers (mean over seeds)")

    for r in rows:
        ax.annotate(f"W={r['w_dose']:.2f}",
                    (r["mean_dose_mean"], r["control_rate_mean"]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color="#a4243b")

    ax.set_xlabel("mean cumulative dose (averaged over seeds)")
    ax.set_ylabel("control rate on unseen patients (%)")
    ax.set_title("Aggregated control-dose frontier\n"
                 "averaged over 5 seeds per operating point")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    results, paths = load_results()
    print(f"Read {len(results)} result files from hpc/results/.")

    grouped = group_by_w_dose(results)
    rows = summarise(grouped)

    # Console table.
    print()
    print(f"{'w_dose':>8} {'seeds':>6} {'control rate (%)':>20} {'mean dose':>18}")
    print("-" * 56)
    for r in rows:
        cr = f"{r['control_rate_mean']:.1f} +/- {r['control_rate_std']:.1f}"
        md = f"{r['mean_dose_mean']:.2f} +/- {r['mean_dose_std']:.2f}"
        print(f"{r['w_dose']:>8.2f} {r['n_seeds']:>6} {cr:>20} {md:>18}")
    print()

    # Warn if any w_dose has fewer than the expected 5 seeds.
    incomplete = [r for r in rows if r["n_seeds"] < 5]
    if incomplete:
        print("Note: some operating points have fewer than 5 seeds. "
              "The sweep may be partial.")
        print()

    json_path = os.path.join(REPORTS_DIR, "frontier_aggregated.json")
    with open(json_path, "w") as f:
        json.dump({"n_results": len(results), "frontier": rows}, f, indent=2)

    png_path = os.path.join(REPORTS_DIR, "frontier_aggregated.png")
    plot_frontier(rows, png_path)

    print("Saved:")
    print(f"  {json_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
