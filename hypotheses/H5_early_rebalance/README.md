# Hypothesis 5 — Early-only rebalancing

**Question.** The reference paper says the shared embedding is dominated by the
output gradient *early in training*. If the bias is set in the first few hundred
steps, then rebalancing only needs to happen **early** — continuing to rebalance
for all 1000 steps may be what over-rotates the embedding and costs keyword
satisfaction.

**Intervention.** Apply the plain input scaling (alpha = 10) only for the first
T steps, then revert to the tied gradient:

    g_eff = alpha * g_in + g_out   (step < T),   else g_in + g_out

**Configs screened.** `early_rebalance:10@250`, `early_rebalance:10@500` (4 seeds).

**"Promising" means.** NLL benefit comparable to full alpha=10 (1.5050) but with
satisfaction closer to tied (0.335), i.e. the trade-off is reduced by scheduling.
