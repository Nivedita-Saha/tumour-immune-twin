"""
Mechanistic tumour-immune model (de Pillis & Radunskaya, 2003).

This is the "virtual patient": the ground-truth dynamics that the neural
surrogate will later learn to imitate, and that the controller will learn
to steer.

State vector (all populations normalised to fractions of carrying capacity):
    N : healthy host cell population
    T : tumour cell population
    I : immune (effector) cell population
    u : drug concentration in the body

Control input:
    v : drug injection rate chosen by the controller (not the drug level
        itself). The body accumulates and clears the drug, so u responds
        to v with a delay. This is what makes steering a real control
        problem rather than a direct assignment.

Reference:
    de Pillis, L.G. and Radunskaya, A., 2003. The dynamics of an optimally
    controlled tumor model: a case study. Mathematical and Computer
    Modelling, 37(11), pp.1221-1244.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Params:
    """
    Parameters of the tumour-immune model.

    IMPORTANT: these are the commonly cited baseline values for this model.
    Verify each one against the parameter table in the source paper before
    reporting any result that depends on them. Treat them as a starting
    point, not as gospel.

    Growth terms
        r1 : tumour growth rate
        r2 : healthy cell growth rate
        b1 : inverse carrying capacity for tumour cells
        b2 : inverse carrying capacity for healthy cells

    Interaction terms
        c1 : rate at which immune cells are exhausted by the tumour
        c2 : rate at which immune cells kill tumour cells
        c3 : competition, healthy cells suppressing tumour cells
        c4 : competition, tumour cells suppressing healthy cells

    Immune terms
        s     : constant baseline supply of immune cells
        rho   : maximum immune recruitment rate in response to the tumour
        alpha : tumour level at which recruitment reaches half its maximum
        d1    : natural death rate of immune cells

    Drug terms
        a1 : drug damage to immune cells
        a2 : drug kill rate on tumour cells
        a3 : drug damage to healthy cells
        d2 : rate at which the body clears the drug
    """

    # growth
    r1: float = 1.5
    r2: float = 1.0
    b1: float = 1.0
    b2: float = 1.0

    # interactions
    c1: float = 1.0
    c2: float = 0.5
    c3: float = 1.0
    c4: float = 1.0

    # immune
    s: float = 0.33
    rho: float = 0.01
    alpha: float = 0.3
    d1: float = 0.2

    # drug
    a1: float = 0.2
    a2: float = 0.3
    a3: float = 0.1
    d2: float = 1.0

    def as_dict(self):
        """Return parameters as a plain dictionary, useful for logging."""
        return asdict(self)


def drug_effect(u):
    """
    Saturating drug effect.

    Returns 1 - exp(-u), which is near zero when there is no drug present
    and approaches 1 at high concentration. The saturation matters: doubling
    the dose does not double the killing, so there is a real cost to
    overtreatment with no matching benefit.
    """
    return 1.0 - np.exp(-u)


def dynamics(t, y, params: Params, v: float = 0.0):
    """
    Right-hand side of the ODE system: how fast each quantity is changing.

    This is the function scipy's solve_ivp will call repeatedly to step the
    system forward in time.

    Args:
        t      : current time. Not used explicitly, but solve_ivp requires
                 it in the signature.
        y      : current state, array-like of [N, T, I, u].
        params : Params instance holding the model constants.
        v      : drug injection rate, the control input. Constant zero means
                 no treatment.

    Returns:
        list of the four derivatives [dN/dt, dT/dt, dI/dt, du/dt].
    """
    N, T, I, u = y
    p = params

    # How strongly the drug is acting right now.
    kill = drug_effect(u)

    # Healthy cells: logistic growth, suppressed by the tumour,
    # damaged by the drug.
    dN = (
        p.r2 * N * (1.0 - p.b2 * N)
        - p.c4 * T * N
        - p.a3 * kill * N
    )

    # Tumour cells: logistic growth, killed by immune cells, suppressed by
    # competition with healthy cells, killed by the drug.
    dT = (
        p.r1 * T * (1.0 - p.b1 * T)
        - p.c2 * I * T
        - p.c3 * T * N
        - p.a2 * kill * T
    )

    # Immune cells: constant supply, recruitment in response to the tumour
    # (saturating), exhaustion from fighting it, natural death, drug damage.
    dI = (
        p.s
        + (p.rho * I * T) / (p.alpha + T)
        - p.c1 * I * T
        - p.d1 * I
        - p.a1 * kill * I
    )

    # Drug concentration: rises with the injection rate, decays as the body
    # clears it. This is why v (what you control) is not the same as u.
    du = v - p.d2 * u

    return [dN, dT, dI, du]


def default_initial_state():
    """
    A starting state representing a patient with an established tumour.

    Healthy tissue is intact, a tumour is present, the immune system is at
    its baseline level, and no drug has been given yet.

    These values are provisional. Step A1.2 will check whether this state
    actually leads to tumour escape when untreated, which is what the
    project needs it to do.
    """
    N0 = 1.0   # healthy tissue at full capacity
    T0 = 0.25  # established tumour
    I0 = 0.15  # baseline immune presence
    u0 = 0.0   # no drug yet
    return np.array([N0, T0, I0, u0], dtype=float)


if __name__ == "__main__":
    # A quick self-check. Running this file directly should print the rate
    # of change of each quantity at the starting state, with no treatment.
    #
    # What to look for: dT/dt should be positive, meaning the tumour is
    # growing when left alone. If it is not, the starting state or the
    # parameters need revisiting before moving on to A1.2.

    p = Params()
    y0 = default_initial_state()
    rates = dynamics(t=0.0, y=y0, params=p, v=0.0)

    labels = ["dN/dt (healthy)", "dT/dt (tumour) ", "dI/dt (immune) ", "du/dt (drug)   "]

    print("Initial state:")
    print(f"  N = {y0[0]:.4f}   T = {y0[1]:.4f}   I = {y0[2]:.4f}   u = {y0[3]:.4f}")
    print()
    print("Rates of change at t = 0, with no treatment (v = 0):")
    for label, rate in zip(labels, rates):
        direction = "growing" if rate > 0 else "shrinking" if rate < 0 else "steady"
        print(f"  {label} = {rate:+.4f}   ({direction})")
    print()
    print("Model loaded successfully.")
