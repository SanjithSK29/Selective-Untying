# Hypothesis 4 — Conflict orthogonalization

**Question.** Every intervention so far *reweights* the input/output gradient
ratio, which also changes magnitudes and can over-rotate. What if the harm is
specifically the **opposition** (negative cosine) between g_in and g_out, and the
clean fix is to surgically *remove* the anti-aligned component of g_out — without
touching its magnitude or the aligned part?

**Intervention.** Project out the component of g_out that points *against* g_in:

    g_eff = g_in + g_out - lam * max(0, -(g_out . g_in_hat)) * g_in_hat

For lam = 1 this makes the output-path gradient orthogonal to (never opposing)
the input-path direction.

**Configs screened.** `orthogonalize:0.5`, `orthogonalize:1.0` (4 seeds).

**"Promising" means.** Improves NLL (toward 1.5050) while keeping satisfaction
at or above tied (0.335) — i.e. de-conflicts without the satisfaction cost.
