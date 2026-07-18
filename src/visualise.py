"""
Publication-quality visualisation of the model's bistable structure.

Purpose (step A1.2, extended): the line plots showed that small tumours are
controlled and large ones escape. These figures make the underlying structure
visible, and quantify it.

Three figures are produced:

  1. Basin map      - tumour burden over time across a fine sweep of starting
                      tumour sizes, drawn as a heatmap. The two basins appear
                      as a hard colour split.

  2. Phase plane    - trajectories drawn in tumour versus healthy tissue space
                      instead of against time. The two stable end states appear
                      as points, and the project's goal becomes visible: move a
                      patient from the diseased attractor to the healthy one.

  3. Separatrix     - final tumour burden as a function of starting tumour
                      burden. The dividing line appears as a near vertical
                      cliff, located precisely by bisection.

Run with:
    python src/visualise.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import Params, default_initial_state
from simulate import simulate, classify


FIGURE_DIR = "figures"

# Outcome colours, used consistently across all figures.
COLOUR_CONTROLLED = "#1b7a5a"   # green, immune system wins
COLOUR_ESCAPE = "#a4243b"       # red, tumour wins
COLOUR_ACCENT = "#0f4c5c"       # dark teal, annotations


def set_style():
    """
    Set plot styling by hand rather than using a named style sheet.

    Named styles get renamed between matplotlib versions, which breaks
    scripts. Setting the values directly is more verbose but will not
    stop working.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.size": 10,
    })


# ----------------------------------------------------------------------
# Computation
# ----------------------------------------------------------------------

def sweep_initial_tumours(T0_values, t_end=100.0, n_points=600):
    """
    Simulate an untreated patient for each starting tumour size.

    Returns:
        t        : shared time array.
        T_matrix : array of shape (len(T0_values), n_points), tumour over time.
        N_matrix : same shape, healthy tissue over time.
        outcomes : list of outcome labels.
    """
    base = default_initial_state()
    params = Params()

    T_rows, N_rows, outcomes = [], [], []

    for T0 in T0_values:
        y0 = base.copy()
        y0[1] = T0
        t, Y = simulate(y0, params=params, v=0.0, t_end=t_end, n_points=n_points)
        N, T = Y[0], Y[1]

        T_rows.append(T)
        N_rows.append(N)
        outcomes.append(classify(T, N))

    return t, np.array(T_rows), np.array(N_rows), outcomes


def find_separatrix(low=0.10, high=0.30, tolerance=1e-4, t_end=100.0):
    """
    Locate the dividing line between the two basins by bisection.

    Bisection works by repeatedly halving an interval known to contain the
    boundary. We start with a tumour size that is controlled and one that
    escapes, then keep testing the midpoint and discarding whichever half
    does not contain the boundary.

    Args:
        low       : a starting tumour size known to be controlled.
        high      : a starting tumour size known to escape.
        tolerance : stop once the bracket is narrower than this.

    Returns:
        The critical starting tumour size, and the number of iterations used.
    """
    base = default_initial_state()
    params = Params()

    def escapes(T0):
        y0 = base.copy()
        y0[1] = T0
        t, Y = simulate(y0, params=params, v=0.0, t_end=t_end, n_points=400)
        return classify(Y[1], Y[0]) == "escape"

    # Confirm the starting bracket actually contains the boundary.
    if escapes(low):
        raise ValueError(f"Lower bound T0 = {low} already escapes. Lower it.")
    if not escapes(high):
        raise ValueError(f"Upper bound T0 = {high} does not escape. Raise it.")

    iterations = 0
    while (high - low) > tolerance:
        mid = 0.5 * (low + high)
        if escapes(mid):
            high = mid
        else:
            low = mid
        iterations += 1

    return 0.5 * (low + high), iterations


# ----------------------------------------------------------------------
# Figure 1: basin map
# ----------------------------------------------------------------------

def plot_basin_map(t, T0_values, T_matrix, separatrix=None):
    """
    Heatmap of tumour burden, with starting tumour size on the vertical axis
    and time on the horizontal axis.

    Each horizontal line across the image is one patient. The sharp colour
    change partway up the image is the boundary between the two basins.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    mesh = ax.pcolormesh(
        t, T0_values, T_matrix,
        cmap="magma", shading="auto", vmin=0.0, vmax=0.6,
    )

    if separatrix is not None:
        ax.axhline(
            separatrix, color="white", linestyle="--", linewidth=1.6,
            label=f"separatrix, T0* = {separatrix:.4f}",
        )
        ax.legend(loc="upper right", labelcolor="white")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("tumour population T")

    ax.set_xlabel("time")
    ax.set_ylabel("starting tumour size  T0")
    ax.set_title("Basins of attraction: one patient per horizontal line")
    ax.grid(False)

    # Explain the two regions directly on the image.
    ax.text(
        t[-1] * 0.55, T0_values[0] + 0.012,
        "immune control\ntumour eliminated",
        color="white", fontsize=9, va="bottom",
    )
    ax.text(
        t[-1] * 0.55, T0_values[-1] - 0.012,
        "tumour escape\nchronic disease state",
        color="white", fontsize=9, va="top",
    )

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "a1_2_basin_map.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Figure 2: phase plane
# ----------------------------------------------------------------------

def plot_phase_plane(T_matrix, N_matrix, outcomes, T0_values):
    """
    Trajectories drawn in tumour versus healthy tissue space.

    Time is not an axis here. Each curve is the path a patient traces
    through the space of possible states, and the point each curve settles
    on is a stable end state of the system.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for T_series, N_series, outcome in zip(T_matrix, N_matrix, outcomes):
        colour = COLOUR_CONTROLLED if outcome == "controlled" else COLOUR_ESCAPE
        ax.plot(T_series, N_series, color=colour, linewidth=1.0, alpha=0.55)
        # Mark where each patient started.
        ax.plot(T_series[0], N_series[0], "o", color=colour, markersize=3.5, alpha=0.9)

    # Mark the two stable end states, taken from where trajectories settle.
    controlled_idx = [i for i, o in enumerate(outcomes) if o == "controlled"]
    escape_idx = [i for i, o in enumerate(outcomes) if o == "escape"]

    if controlled_idx:
        i = controlled_idx[0]
        ax.plot(T_matrix[i][-1], N_matrix[i][-1], "*", color=COLOUR_CONTROLLED,
                markersize=22, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(
            "healthy state\ntumour cleared,\ntissue restored",
            xy=(T_matrix[i][-1], N_matrix[i][-1]),
            xytext=(0.10, 0.90), fontsize=9, color=COLOUR_CONTROLLED,
            arrowprops=dict(arrowstyle="->", color=COLOUR_CONTROLLED, linewidth=1.1),
        )

    if escape_idx:
        i = escape_idx[-1]
        ax.plot(T_matrix[i][-1], N_matrix[i][-1], "*", color=COLOUR_ESCAPE,
                markersize=22, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(
            "diseased state\ntumour persists,\ntissue depleted",
            xy=(T_matrix[i][-1], N_matrix[i][-1]),
            xytext=(0.30, 0.50), fontsize=9, color=COLOUR_ESCAPE,
            arrowprops=dict(arrowstyle="->", color=COLOUR_ESCAPE, linewidth=1.1),
        )

    ax.set_xlabel("tumour population  T")
    ax.set_ylabel("healthy cell population  N")
    ax.set_title("Phase plane: two stable end states, and the paths into them")

    handles = [
        Line2D([0], [0], color=COLOUR_CONTROLLED, linewidth=2, label="immune control"),
        Line2D([0], [0], color=COLOUR_ESCAPE, linewidth=2, label="tumour escape"),
        Line2D([0], [0], marker="o", color="grey", linestyle="none",
               markersize=5, label="starting state"),
        Line2D([0], [0], marker="*", color="grey", linestyle="none",
               markersize=12, label="stable end state"),
    ]
    ax.legend(handles=handles, loc="lower left")

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "a1_2_phase_plane.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Figure 3: separatrix
# ----------------------------------------------------------------------

def plot_separatrix(T0_values, T_matrix, outcomes, separatrix):
    """
    Final tumour burden plotted against starting tumour burden.

    A gradual relationship would give a smooth curve. Bistability gives a
    cliff: a vanishingly small change in the starting tumour size flips the
    outcome completely.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    final_T = T_matrix[:, -1]
    colours = [
        COLOUR_CONTROLLED if o == "controlled" else COLOUR_ESCAPE
        for o in outcomes
    ]

    ax.scatter(T0_values, final_T, c=colours, s=26, zorder=3)
    ax.plot(T0_values, final_T, color="#999999", linewidth=0.8, zorder=2)

    ax.axvline(separatrix, color=COLOUR_ACCENT, linestyle="--", linewidth=1.5, zorder=1)
    ax.annotate(
        f"separatrix\nT0* = {separatrix:.4f}",
        xy=(separatrix, 0.30),
        xytext=(separatrix + 0.035, 0.30),
        fontsize=9.5, color=COLOUR_ACCENT, va="center",
        arrowprops=dict(arrowstyle="->", color=COLOUR_ACCENT, linewidth=1.1),
    )

    ax.set_xlabel("starting tumour size  T0")
    ax.set_ylabel("final tumour size  T at end of horizon")
    ax.set_title("A knife edge: the outcome flips at a single critical tumour size")

    handles = [
        Line2D([0], [0], marker="o", color=COLOUR_CONTROLLED, linestyle="none",
               markersize=7, label="controlled"),
        Line2D([0], [0], marker="o", color=COLOUR_ESCAPE, linestyle="none",
               markersize=7, label="escape"),
    ]
    ax.legend(handles=handles, loc="center right")

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "a1_2_separatrix.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------

if __name__ == "__main__":
    set_style()

    print("Locating the separatrix by bisection...")
    separatrix, iterations = find_separatrix()
    print(f"  critical starting tumour size T0* = {separatrix:.5f}")
    print(f"  found in {iterations} iterations\n")

    # Sample finely around the separatrix, coarsely away from it, so the
    # interesting region is well resolved without wasting computation.
    T0_values = np.unique(np.concatenate([
        np.linspace(0.02, separatrix - 0.02, 22),
        np.linspace(separatrix - 0.02, separatrix + 0.02, 26),
        np.linspace(separatrix + 0.02, 0.55, 22),
    ]))

    print(f"Simulating {len(T0_values)} untreated patients...")
    t, T_matrix, N_matrix, outcomes = sweep_initial_tumours(T0_values)

    n_controlled = sum(1 for o in outcomes if o == "controlled")
    n_escape = sum(1 for o in outcomes if o == "escape")
    n_other = len(outcomes) - n_controlled - n_escape
    print(f"  controlled: {n_controlled}    escape: {n_escape}    other: {n_other}\n")

    p1 = plot_basin_map(t, T0_values, T_matrix, separatrix=separatrix)
    p2 = plot_phase_plane(T_matrix, N_matrix, outcomes, T0_values)
    p3 = plot_separatrix(T0_values, T_matrix, outcomes, separatrix)

    print("Saved:")
    for p in (p1, p2, p3):
        print(f"  {p}")
    print()
    print(f"Headline result: the model is bistable, with the boundary between")
    print(f"immune control and tumour escape at T0* = {separatrix:.4f}.")
    print("The control problem is therefore to push a patient across this")
    print("boundary, after which the immune system completes the job unaided.")
