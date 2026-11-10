# Five exploratory hypotheses

Five mechanistically distinct follow-ups to the null result, each screened with
4 seeds (≈27 min each) to judge promise before any full 12-seed run.

| # | Folder | Intervention | Axis |
|---|---|---|---|
| H1 | `H1_inverse_conflict` | rebalance *aligned* (low-conflict) tokens | which tokens |
| H2 | `H2_norm_scaling` | rebalance by output-gradient *magnitude* | which tokens |
| H3 | `H3_output_damping` | damp the output gradient (gamma < 1) | how (mirror) |
| H4 | `H4_orthogonalize` | remove the anti-aligned g_out component | how (surgical) |
| H5 | `H5_early_rebalance` | input scaling only early in training | when |

**Shared harness.** `screen.py` (one script, five interventions). Each folder
has its own `run.sh` and `README.md`; results land in `H*/results.json`.

**Reference points (4 seeds, from committed results):**
- tied: satisfaction ≈ 0.335, NLL ≈ 1.508
- global input scaling alpha=10: satisfaction ≈ 0.327, NLL ≈ 1.505

A hypothesis is "promising" if it moves NLL toward 1.505 **without** the
satisfaction drop to 0.327 (or reveals a clean, distinct effect).
