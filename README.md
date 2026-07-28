# Tumour-Immune Digital Twin

A controllable digital twin of tumour-immune dynamics. A neural surrogate is learned from a mechanistic virtual-patient cohort, then a reinforcement-learning controller is wrapped around that surrogate to compute drug dosing that steers virtual patients from a tumour-escape state toward a controlled, healthy equilibrium.

The project combines three capabilities in one system: deep learning (the surrogate), control (the dosing policy), and the modelling of biological systems (the mechanistic ground truth).

## The idea

Cancer can be read as a breakdown in the balance between three cell populations: tumour cells, immune effector cells, and healthy host cells. A digital twin is a computational model of that system which can be both simulated and steered, so candidate interventions can be tested in silico before the lab or clinic.

Two research traditions each solve half of this and leave a gap:

- **Mechanistic ODE models** (de Pillis and Radunskaya, 2003; Kirschner and Panetta, 1998) describe tumour-immune dynamics well and are controllable, but they are hand-built and fixed: their equations and parameters are set in advance, they are hard to adapt to new data, and re-solving the control problem for each new scenario is costly.
- **Neural ODEs and learned surrogates** (Chen et al., 2018) fit dynamics from data and stay smooth and differentiable, but they are not built for control.

There is no readily available model that is at once learned from data and directly controllable. This project builds one, and asks a specific question: can a learned surrogate reproduce mechanistic tumour-immune dynamics accurately enough that a controller trained on it steers unseen virtual patients to tumour control more efficiently than a fixed-dose schedule?

## How it works

The system is built as a pipeline of independent, validated stages.

**1. Mechanistic virtual patient (`src/model.py`).** The de Pillis-Radunskaya ODE model is the ground-truth dynamics, with drug injection rate exposed as the control input. The body accumulates and clears the drug, so the drug level responds to the injection rate with a delay. This is what makes dosing a genuine control problem rather than a direct assignment.

**2. Virtual-patient cohort (`src/cohort.py`, `src/generate_data.py`).** Physiological parameters and initial conditions are varied by Latin hypercube sampling to produce a cohort of virtual patients, each labelled by its untreated outcome (escape, controlled, host failure, or intermediate). This yields the (state, input, next-state) trajectory data the surrogate learns from, with a strict patient-level train/test split (`src/make_splits.py`) so no controller is ever evaluated on a patient it practised on.

**3. Neural surrogate (`src/train_surrogate.py`, `src/model.py`).** A neural one-step predictor is trained to imitate the mechanistic dynamics, with a Neural ODE formulation available as an upgrade. The payoff is speed: reinforcement learning needs millions of environment steps, and integrating the ODE that many times would take days, whereas the surrogate produces a step in a single network evaluation.

**4. Control environment (`src/env.py`).** The surrogate is wrapped as a Gymnasium environment. The reward penalises both tumour burden and drug administered. The dose penalty is the important half: without it the agent would simply administer maximum dose forever, which does control the tumour but is exactly the wasteful behaviour the project sets out to improve on. Terminal signals add a large penalty for host-tissue collapse and a bonus for reaching and holding tumour control.

**5. Learned controller (`src/train_controller.py`, `src/frontier.py`).** A PPO policy is trained on the surrogate environment, then a sweep over the dose penalty produces a family of controllers tracing the trade-off between how many patients are controlled and how much drug is used.

Outcome definitions ("controlled", "escape", "host failure") are fixed in advance in `metrics_note.md`, before any controller is trained, so results are judged against a standard set rather than one chosen to flatter them.

## What the results show

### The surrogate is faithful

Validated on 120 rollouts across 40 unseen test patients, under untreated, constant, and randomised dosing:

- Normalised rollout error over a full horizon is on the order of 0.001 across all state variables and dosing schedules.
- Outcome agreement with the mechanistic model is 119 of 120 (99.2 per cent).
- The separatrix, the tumour level dividing control from escape, is matched to within 0.0024 (mechanistic 0.155, surrogate 0.153).
- Long rollouts stay bounded and physically plausible.

Rollout error matters far more than one-step error here, because the controller acts over the full horizon, and the surrogate holds up over the full horizon. This validates using it in place of the ODE for control.

### The learned controller dominates fixed dosing, but biology sets a ceiling

Evaluated on held-out test patients, the family of learned controllers sits above and to the left of the constant-dose baseline curve: for a given amount of drug it controls more patients, and for a given control rate it uses less drug. For example, a learned controller reaches a control rate comparable to the constant-0.7 schedule while using well under half the cumulative dose, and another matches the maximum-dose control rate at roughly 75 per cent of the drug. Because this holds across the whole operating range rather than at one convenient setting, it is a far more robust claim than any single comparison.

The headline target set at the outset, controlling at least 80 per cent of patients, was **not** met, and the reason is the most interesting result in the project. It is structural, not a failure of the controller. Around 70 per cent of the patients that need rescue are monostable: no healthy attractor exists for them, so no dosing strategy of any kind can save them. The achievable control rate is capped by the biology of the cohort, not by the policy. Among structurally treatable patients the learned controller is the strongest option tested. Distinguishing "can this controller steer?" from "is this patient steerable at all?" is exactly the distinction an honest evaluation has to make.

## Run it yourself

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/generate_data.py      # build the virtual-patient cohort
python src/make_splits.py        # patient-level train/test split
python src/train_surrogate.py    # train and validate the neural surrogate
python src/train_controller.py   # train the PPO dosing policy
python src/frontier.py           # sweep the control-versus-dose frontier
python src/visualise.py          # produce trajectory and frontier plots
```

## What this demonstrates

- A complete scientific-machine-learning pipeline: mechanistic model, data generation, learned surrogate, and control, each stage validated before the next is built on it.
- Neural surrogate modelling of continuous dynamical systems, with rollout stability and outcome fidelity treated as first-class metrics.
- Reinforcement-learning control with a reward designed to make the central claim (equal-or-better control using less drug) testable rather than assumed.
- Rigorous evaluation practice: outcome definitions fixed in advance, strict held-out patient splits, comparison against no-treatment, constant-dose, and maximum-dose baselines, and honest separation of a controller's performance from the structural controllability of the problem.

## References (Harvard)

Chen, R.T.Q., Rubanova, Y., Bettencourt, J. and Duvenaud, D., 2018. Neural ordinary differential equations. *Advances in Neural Information Processing Systems*, 31.

de Pillis, L.G. and Radunskaya, A., 2003. The dynamics of an optimally controlled tumor model: a case study. *Mathematical and Computer Modelling*, 37(11), pp.1221-1244.

Kirschner, D. and Panetta, J.C., 1998. Modeling immunotherapy of the tumor-immune interaction. *Journal of Mathematical Biology*, 37(3), pp.235-252.
