# Hypothesis 2 — Gradient-magnitude scaling

**Question.** Conflict is *directional* (input vs output gradient angle) and is
orthogonal to gradient *magnitude* (rho ~ 0). Maybe the relevant axis is not the
angle but **how hard the output path pulls each token** (|g_out|). Tokens whose
output gradient is largest are the most "stolen"; rebalancing those may matter
more than the conflict-ranked ones.

**Intervention.** Scale the input-path gradient by output-gradient magnitude:

    g_eff,i = (1 + beta * w_i) * g_in,i + g_out,i,    w_i = |g_out_i| / max|g_out|

**Configs screened.** `norm_scaling:5`, `norm_scaling:10` (4 seeds).

**"Promising" means.** A clean NLL benefit (toward 1.5050) with satisfaction
held closer to tied (0.335) than the conflict-selected weighting achieved.
