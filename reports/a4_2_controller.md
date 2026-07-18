# Learned Controller (A4.2)

PPO trained for 400,000 steps on 216 treatable training patients.
Evaluated on 46 held-out test patients.

| policy | control rate | mean dose | mean reward |
|---|---|---|---|
| no treatment | 0.0 % | 0.00 | -57.93 |
| constant v = 0.3 | 13.0 % | 30.00 | -45.93 |
| constant v = 0.5 | 17.4 % | 50.00 | -41.81 |
| maximum dose | 32.6 % | 100.00 | -36.01 |
| treat until T < 0.10 | 17.4 % | 45.07 | -41.09 |
| **learned controller** | **32.6 %** | **79.66** | **-33.45** |

## Notes

- Patients are structurally treatable only: a healthy attractor exists for them. Around 70 percent of patients needing rescue are monostable and cannot be helped by any dose.
- Host failure did not occur under any policy in this population, so the safety penalty in the reward was never triggered. The problem is control rate against dose economy.
