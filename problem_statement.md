# Problem Statement: A Controllable Digital Twin of Tumour-Immune Cell Dynamics

**Project:** `tumour-immune-twin`
**Context:** Portfolio project for the KTH doctoral position *Deep Learning for Biological Systems* (Decision & Control Systems, KTH)
**Author:** Nivedita Saha
**Deliverable:** A0.1

---

## 1. Background and motivation

Cancer can be understood as a breakdown of the normal balance between cell populations: tumour cells, immune (effector) cells, and healthy host cells. The immune system can naturally hold a tumour in check, but tumours often escape this control and grow. Restoring a healthy balance through drug development is still largely a process of trial and error, which is slow and expensive.

A **digital twin**, a computational model of a specific biological system that can be simulated and *steered*, offers a way to test interventions *in silico* before committing to the lab or clinic. If the twin is both realistic and controllable, we can search for dosing strategies that push the system back toward a healthy state, reducing trial and error and speeding up the path from discovery to treatment.

## 2. What existing models give us

Mechanistic ordinary differential equation (ODE) models already describe tumour-immune dynamics well:

- **Kirschner & Panetta (1998)** model the interaction between effector immune cells, tumour cells, and the signalling molecule IL-2, with immunotherapy as an external input. They show how treatment can move the system between three outcomes: uncontrolled tumour growth, tumour dormancy, or tumour clearance.
- **de Pillis & Radunskaya (2003)** extend this picture to include healthy host cells and a drug term, and apply **optimal control** to schedule drug dosing so the tumour is eliminated while healthy tissue is spared. This is an early, explicit example of *steering* the system toward a healthy equilibrium.

These models are interpretable, but they are **hand-built and fixed**. Their equations and parameters are specified in advance, they are hard to adapt to new data, and re-solving the optimal-control problem for every new scenario is computationally costly.

Separately, **Chen et al. (2018)** introduced **Neural Ordinary Differential Equations**, where a neural network learns the *rate of change* of a system and an ODE solver integrates it forward in time. This yields a model that is **learned from data** yet remains continuous, smooth, and differentiable. These properties make it well suited to being embedded inside a controller.

## 3. The gap

There is currently no readily available model of tumour-immune dynamics that is at once **(a)** learned from data rather than fully hand-specified, and **(b)** directly controllable, so that interventions can be computed to drive the system toward a healthy equilibrium. Mechanistic models are controllable but rigid; standard machine-learning models fit data but are not built for control.

## 4. Aim and research question

**Aim.** Build a *controllable digital twin* of tumour-immune dynamics: learn a neural surrogate of the dynamics from a mechanistic virtual-patient cohort, then wrap a controller around it that computes drug / immunotherapy dosing to steer virtual patients from a tumour-escape state to a controlled, healthy equilibrium.

**Research question.** *Can a learned neural surrogate reproduce mechanistic tumour-immune dynamics accurately enough that a controller trained on it steers unseen virtual patients to tumour control more efficiently than a fixed-dose schedule?*

## 5. Objectives

- **O1.** Implement a trusted mechanistic model (de Pillis-Radunskaya as the primary model; Kirschner-Panetta as a simpler fallback) as the ground-truth "virtual patient", with drug / therapy exposed as a control input.
- **O2.** Generate a **virtual-patient cohort** by varying physiological parameters, producing (state, input, next-state) trajectory data.
- **O3.** Train a **neural surrogate** (an MLP one-step predictor first, a Neural ODE as an upgrade) and validate it against the mechanistic model.
- **O4.** Wrap the surrogate as a control environment and **learn a dosing policy** (reinforcement learning; model predictive control as a control-theoretic comparison).
- **O5.** Evaluate efficacy, efficiency, robustness, and generalisation, and package the twin with a short interactive demo.

## 6. Success metrics

**Primary (headline) metric**

> The learned controller drives **≥ 80 %** of held-out virtual patients from a tumour-escape state to a tumour-controlled equilibrium within the treatment horizon, using **less cumulative drug** than a fixed-dose baseline.

**Supporting metrics**

| Metric | Target |
|---|---|
| Surrogate fidelity | Multi-step trajectory error vs the mechanistic model < ~5-10 % (normalised) |
| Tumour reduction | Final tumour burden reduced by ≥ 90 % vs the untreated trajectory |
| Dose economy | Cumulative dose lower than a constant / maximal-dose schedule for equal-or-better outcome |
| Robustness | Performance retained under observation noise and parameter perturbation |
| Generalisation | Metrics hold on virtual patients not seen during training |

*(Targets are provisional and will be finalised once the baseline is implemented.)*

## 7. Relevance to the position

The project mirrors the research group's stated goals, building and validating digital twins of cellular interactions and steering cellular systems toward healthier states, and combines the three capabilities the advert emphasises: **deep learning, control, and the modelling of biological systems.**

---

## References (Harvard)

Chen, R.T.Q., Rubanova, Y., Bettencourt, J. and Duvenaud, D., 2018. Neural ordinary differential equations. *Advances in Neural Information Processing Systems*, 31.

de Pillis, L.G. and Radunskaya, A., 2003. The dynamics of an optimally controlled tumor model: a case study. *Mathematical and Computer Modelling*, 37(11), pp.1221-1244.

Kirschner, D. and Panetta, J.C., 1998. Modeling immunotherapy of the tumor-immune interaction. *Journal of Mathematical Biology*, 37(3), pp.235-252.
