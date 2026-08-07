# Surrogate Validation (A3.3)

Rollouts: 120 across 40 unseen test patients, under untreated, constant and randomised dosing.

## Rollout error over a full horizon

| schedule | N healthy | T tumour | I immune | u drug |
|---|---|---|---|---|
| untreated | 0.00020 | 0.00016 | 0.00044 | 0.00014 |
| constant | 0.00025 | 0.00023 | 0.00034 | 0.00017 |
| randomised | 0.00024 | 0.00022 | 0.00041 | 0.00018 |

## Outcome agreement

120 of 120 (100.0 %)

## Separatrix

- Mechanistic model: T0* = 0.15503
- Surrogate: T0* = 0.15605
- Difference: 0.00102

## Physical plausibility

- Most negative value: -0.00149
- Largest value: 2.41665
