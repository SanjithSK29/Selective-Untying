# Results from committed aggregate records

These calculations do not independently reproduce training or evaluation.

| Method | Seeds | Satisfaction (%) | Sample SD (pp) | Keyword recall (%) | Response NLL |
|---|---:|---:|---:|---:|---:|
| tied | 12 | 33.53 | 2.17 | 70.48 | 1.50832 |
| conflict | 12 | 33.68 | 2.20 | 70.54 | 1.50973 |
| frequency | 12 | 33.72 | 1.73 | 70.44 | 1.50801 |
| freq_random | 12 | 33.80 | 2.17 | 70.32 | 1.50925 |

## Paired differences

| Comparison | Satisfaction difference (pp) | Approx. 95% t interval (pp) |
|---|---:|---:|
| conflict_minus_frequency | -0.03 | [-0.92, +0.85] |
| conflict_minus_freq_random | -0.12 | [-0.91, +0.67] |
| conflict_minus_tied | +0.15 | [-0.66, +0.96] |

Descriptive paired t intervals over SFT seeds on one fixed evaluation sample and one pretrained checkpoint. Historical inputs are rounded aggregates. No prompt-level or pretraining uncertainty.

The criterion is not met for the historical records. Failure to meet it does not establish equivalence or absence of effects in other settings.

NLL measures teacher-forced reference prediction. Keyword presence measures lexical compliance. Human fluency/coherence has not been established by these files.
