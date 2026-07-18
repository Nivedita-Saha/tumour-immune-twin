# Surrogate Validation (A3.3)

Rollouts: 120 across 40 unseen test patients, under untreated, constant and randomised dosing.

## Rollout error over a full horizon

| schedule | N healthy | T tumour | I immune | u drug |
|---|---|---|---|---|
| untreated | 0.00105 | 0.00103 | 0.00202 | 0.00072 |
| constant | 0.00096 | 0.00092 | 0.00153 | 0.00073 |
| randomised | 0.00094 | 0.00089 | 0.00170 | 0.00077 |

## Outcome agreement

119 of 120 (99.2 %)

## Separatrix

- Mechanistic model: T0* = 0.15503
- Surrogate: T0* = 0.15262
- Difference: 0.00241

## Physical plausibility

- Most negative value: -0.00394
- Largest value: 2.37091
