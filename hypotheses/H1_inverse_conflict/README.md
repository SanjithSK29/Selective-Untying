# Hypothesis 1 — Inverse-conflict gradient scaling

**Question.** The conflict-weighted scaling (rebalancing the *high*-conflict tokens,
i.e. content words / character names) hurt keyword satisfaction while barely
moving NLL. Is the *fluency* benefit of global input-gradient scaling instead
concentrated in the **low / negatively-conflicted** tokens (function words, whose
input and output gradients are already aligned)?

**Intervention.** Scale the input-path gradient only for **aligned** tokens:

    g_eff,i = (1 + beta * w_i) * g_in,i + g_out,i,    w_i = max(0, -c_i) / max

i.e. the mirror image of the earlier conflict weighting.

**Configs screened.** `inverse_conflict:5`, `inverse_conflict:10` (4 seeds).

**"Promising" means.** NLL improves toward the global alpha=10 level (1.5050)
while keyword satisfaction stays near tied (0.335) instead of dropping to 0.327.
