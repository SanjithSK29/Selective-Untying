# Hypothesis 3 — Output-gradient damping

**Question.** The reference paper's claim is that the *output* gradient
dominates the tied embedding. The project so far only **boosted the input**
gradient (alpha > 1). What happens if we instead **damp the output** gradient
directly (gamma < 1)? Under Adam these are directionally equivalent to input
scaling (same alpha/gamma ratio), but gradient clipping and the nonlinear
update make them distinct in practice — testing the mirror reveals whether the
effect is about the *ratio* or the *absolute magnitudes*.

**Intervention.**    g_eff = g_in + gamma * g_out,    gamma < 1.

**Configs screened.** `output_damping:0.5`, `output_damping:0.1` (4 seeds).

**"Promising" means.** Reproduces (or beats) the global input-scaling NLL
benefit at a *smaller* satisfaction cost, i.e. a cleaner trade-off — or a
qualitatively different result that distinguishes ratio from magnitude.
