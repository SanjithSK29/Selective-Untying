# Post-hoc non-overlap sensitivity analysis

The full scan found normalized exact pretraining matches for **129/500 (25.8%)**
evaluation stories. No matches occurred in the reconstructed SFT/calibration
training prefixes. The following analysis uses the remaining **371 prompts**.

| Method | Seeds | Satisfaction (%) | Macro recall (%) |
|---|---:|---:|---:|
| tied | 12 | 33.87 | 70.71 |
| conflict | 12 | 34.05 | 70.81 |
| frequency | 12 | 34.14 | 70.66 |
| freq_random | 12 | 34.14 | 70.54 |

| Conflict minus comparator | Satisfaction difference (pp) | Paired 95% interval (pp) |
|---|---:|---:|
| frequency | -0.09 | [-1.34, +1.16] |
| freq_random | -0.09 | [-1.21, +1.03] |
| tied | +0.18 | [-0.68, +1.04] |

Post-hoc sensitivity analysis on 371 stories with no normalized exact match in scanned training sources. Not a new independent test set; near-duplicate overlap not checked. NLL unavailable.

This subset analysis does not erase contamination in the full-set evaluation.
It was defined after the main runs. Intervals concern SFT seeds on this fixed subset.
The original models and generations were not rerun or changed. Response NLL on
this subset cannot be reconstructed from aggregate NLL and missing-word lists.
No subset NLL or human-coherence score is invented.
