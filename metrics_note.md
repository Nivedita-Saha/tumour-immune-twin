# Metrics Note: Defining "Healthy", "Tumour Escape", and How Success Is Measured

**Project:** `tumour-immune-twin`
**Deliverable:** A0.3
**Status:** Provisional. Thresholds marked (P) are to be finalised after A1.2, once the mechanistic model's baseline behaviour has been reproduced.

---

## 1. Why this note exists

Every later claim in this project depends on two words: "healthy" and "controlled". If those are defined after the results are in, the evaluation is not trustworthy. This note fixes the definitions in advance, so that outcomes are judged against a standard set before any controller is trained.

## 2. State variables

Following de Pillis & Radunskaya (2003), the system state at time *t* is:

| Symbol | Meaning | Units |
|---|---|---|
| `T(t)` | Tumour cell population | Fraction of carrying capacity, normalised to [0, 1] |
| `N(t)` | Healthy host cell population | Fraction of carrying capacity, normalised to [0, 1] |
| `I(t)` | Immune (effector) cell population | Normalised |
| `u(t)` | Drug / therapy input (the control) | Normalised dose rate, bounded [0, u_max] |

The control input `u(t)` is what the controller chooses. Everything else evolves according to the dynamics.

## 3. Outcome definitions

A trajectory is classified at the end of the treatment horizon `T_end`, using a sustained window rather than a single instant, so that a brief dip does not count as success.

### 3.1 Tumour controlled

A trajectory is **tumour controlled** if, for the final 10 % of the horizon:

- `T(t) < T_low` where **T_low = 0.05 (P)**, and
- `T(t)` is non-increasing on average (no late rebound), and
- `N(t) > N_min` where **N_min = 0.50 (P)**.

The third condition is essential: it requires that healthy tissue survives. Without it, a controller could "succeed" by administering maximum dose and destroying the host along with the tumour.

### 3.2 Tumour escape

A trajectory is **tumour escape** if, at `T_end`:

- `T(t) > T_high` where **T_high = 0.40 (P)**, and
- `T(t)` is increasing.

### 3.3 Intermediate / dormant

Anything falling between the two above is recorded as **intermediate** and reported separately, not silently counted as either a success or a failure. Honest reporting of this middle category matters more than a clean-looking headline number.

### 3.4 Host failure

An override condition. If `N(t) < N_fail` where **N_fail = 0.20 (P)** at any point, the trajectory is classified as **host failure** regardless of tumour burden. This is a failure, never a success, even if `T` reached zero.

## 4. Initial conditions

Virtual patients are initialised in a **tumour-escape trajectory**, meaning that with `u(t) = 0` the tumour grows and the outcome would be escape. This is the point of the project: the untreated system is heading somewhere bad, and the controller must redirect it.

The precise initial values are drawn from the parameter cohort in A2.1 and recorded with each patient.

## 5. Evaluation metrics

### 5.1 Primary metric

**Control rate:** the percentage of held-out virtual patients classified as *tumour controlled* (Section 3.1) at the end of the horizon.

Target: **≥ 80 % (P)**.

### 5.2 Supporting metrics

| Metric | Definition | Direction |
|---|---|---|
| Final tumour burden | `T(T_end)`, and reduction relative to the untreated trajectory | Lower is better |
| Time to control | First time `T(t) < T_low` and stays below it for the rest of the horizon | Lower is better |
| Cumulative dose | Integral of `u(t)` over the horizon | Lower is better |
| Healthy tissue preserved | Minimum of `N(t)` across the trajectory | Higher is better |
| Peak tumour burden | Maximum of `T(t)` across the trajectory | Lower is better |

### 5.3 Surrogate fidelity metrics

Separate from control performance, the neural surrogate itself is judged on:

| Metric | Definition |
|---|---|
| One-step error | Mean absolute error of the predicted next state |
| Rollout error | Normalised trajectory error over the full horizon, surrogate vs mechanistic model |
| Rollout stability | Whether long rollouts remain bounded and physically plausible, no negative populations, no divergence |

Rollout error matters far more than one-step error. A model with small one-step error can still drift badly over hundreds of steps, and the controller acts over the full horizon.

## 6. Baselines to compare against

A result means nothing without a comparison. Every controller is reported against:

1. **No treatment** (`u = 0`), which establishes what happens if nothing is done.
2. **Constant dose**, a fixed `u` applied throughout, the simplest non-trivial strategy.
3. **Maximum dose** (`u = u_max`), which shows the cost of overtreatment and usually triggers host failure.
4. **Learned controller** (RL, and MPC at the stretch step).

The headline claim of the project is that the learned controller achieves a comparable or better control rate than the constant-dose baseline **while using less cumulative drug**.

## 7. Reporting rules

- Metrics are reported on **held-out virtual patients only**, never on patients used for training.
- Results are averaged over multiple random seeds, with variation reported, not a single lucky run.
- The intermediate and host-failure categories are always shown, not folded into other numbers.
- Thresholds in this note are fixed before controller training and are not adjusted afterwards.

---

## References (Harvard)

de Pillis, L.G. and Radunskaya, A., 2003. The dynamics of an optimally controlled tumor model: a case study. *Mathematical and Computer Modelling*, 37(11), pp.1221-1244.

Kirschner, D. and Panetta, J.C., 1998. Modeling immunotherapy of the tumor-immune interaction. *Journal of Mathematical Biology*, 37(3), pp.235-252.
