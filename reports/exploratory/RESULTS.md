# Results from committed aggregate records

These calculations do not independently reproduce training or evaluation.

| Method | Seeds | Satisfaction (%) | Sample SD (pp) | Keyword recall (%) | Response NLL |
|---|---:|---:|---:|---:|---:|
| tied | 3 | 34.40 | 4.16 | not recorded | 1.50743 |
| full | 3 | 33.07 | 3.93 | not recorded | 1.50480 |
| conflict | 3 | 34.53 | 3.26 | not recorded | 1.50883 |
| frequency | 3 | 34.07 | 2.81 | not recorded | 1.50707 |
| freq_random | 3 | 34.73 | 3.10 | not recorded | 1.50843 |
| grad_norm | 3 | 34.00 | 4.01 | not recorded | 1.50653 |
| alpha_2 | 3 | 34.20 | 3.75 | not recorded | 1.50683 |
| alpha_5 | 3 | 33.47 | 4.41 | not recorded | 1.50543 |
| alpha_10 | 3 | 32.80 | 3.56 | not recorded | 1.50400 |

## Paired differences

| Comparison | Satisfaction difference (pp) | Approx. 95% t interval (pp) |
|---|---:|---:|
| conflict_minus_frequency | +0.47 | [-5.01, +5.94] |
| conflict_minus_freq_random | -0.20 | [-0.70, +0.30] |
| conflict_minus_tied | +0.13 | [-2.60, +2.87] |
| full_minus_tied | -1.33 | [-2.77, +0.10] |
| alpha_10_minus_tied | -1.60 | [-3.88, +0.68] |

Descriptive paired t intervals over SFT seeds on one fixed evaluation sample and one pretrained checkpoint. Historical inputs are rounded aggregates. No prompt-level or pretraining uncertainty.

The criterion is not met for the historical records. Failure to meet it does not establish equivalence or absence of effects in other settings.

NLL measures teacher-forced reference prediction. Keyword presence measures lexical compliance. Human fluency/coherence has not been established by these files.
