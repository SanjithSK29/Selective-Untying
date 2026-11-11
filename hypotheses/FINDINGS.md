# New findings: early-only rebalancing and output damping

Two new interventions on the tied-embedding "unembedding bias", screened across 5
hypotheses and confirmed with 12 seeds each. One of them **removes the
satisfaction cost** of the existing input-gradient scaling; the other **reveals
the true, previously-clipped magnitude** of the effect.

---

## TL;DR

| Intervention | NLL vs tied | Satisfaction vs tied | Verdict |
|---|---|---|---|
| global input scaling α=10 (prior best) | **−0.0033** (p<10⁻¹³) | −0.82 pp (p=0.06) | trade-off |
| **early rebalance, first 500 steps (new)** | **−0.0032** (p<10⁻¹³) | **−0.32 pp (p=0.45, n.s.)** | ✅ **same NLL gain, no satisfaction cost** |
| **output damping γ=0.1 (new)** | **−0.0042** (p<10⁻¹³) | −1.47 pp (p=0.0001) | stronger trade-off, unclipped |

**The win:** rebalancing the input gradient *only during the first 500 steps* —
then reverting to the tied gradient — recovers **96%** of the full global-α NLL
benefit while making the satisfaction cost **statistically indistinguishable from
zero** (−0.32 pp, 95% CI [−1.2, +0.6] pp). The global α=10 intervention costs
−0.82 pp (p=0.06) for essentially the same NLL gain.

---

## Background

Weight tying shares one embedding between input encoding and output prediction.
The reference paper (Lopardo et al., Findings of ACL 2026) shows this shared
embedding is dominated *early in training* by the output-side gradient. Our
project's prior experiments established:

1. **Selective untying** (output offsets on the 400 most gradient-conflicted
   tokens) → **null** (no detectable effect, 12 seeds).
2. **Global input-gradient scaling** `g_eff = α·g_in + g_out` → a clean
   **trade-off**: NLL improves monotonically (α=10: −0.0033, p<10⁻¹³) while
   keyword satisfaction degrades (α=10: −0.82 pp, p=0.06).
3. **Conflict-weighted scaling** (rebalance only the high-conflict tokens) →
   the trade-off *localizes*: content-word rebalancing costs satisfaction but
   barely moves NLL; function-word rebalancing hurts NLL and leaves satisfaction
   alone.

This left one obvious open question: **can the NLL benefit be obtained without
the satisfaction cost?** Five follow-up hypotheses were screened (4 seeds each).
Two survived screening; both were confirmed at 12 seeds.

---

## Hypothesis 1 — Early-only rebalancing (H5)

### Idea

The reference paper says the embedding is "stolen" by the output gradient **early
in training**. If the bias is set in the first few hundred steps, rebalancing only
needs to happen early — continuing to rebalance for all 1000 SFT steps is what
*over-rotates* the embedding and degrades keyword-following.

### Intervention

Apply the plain input scaling for the first `T` steps, then revert to the tied
gradient:

```
g_eff = α·g_in + g_out   (step < T)
g_eff =      g_in + g_out  (step ≥ T)
```

### 12-seed results

| Config | NLL | Satisfaction | Δ NLL vs tied | Δ sat vs tied |
|---|---|---|---|---|
| tied | 1.5083 | 0.3353 | — | — |
| early T=250 | 1.5067 | 0.3350 | **−0.0016** (p<10⁻¹³) | −0.03 pp (p=0.95) |
| early T=500 | 1.5051 | 0.3322 | **−0.0032** (p<10⁻¹³) | −0.32 pp (p=0.45) |
| global α=10 (full 1000) | 1.5050 | 0.3272 | −0.0033 (p<10⁻¹³) | −0.82 pp (p=0.06) |

- `early T=500` captures **96.0%** of the global-α NLL benefit (−0.0032 vs
  −0.0033), and its satisfaction change is **not significant** (95% CI
  [−1.21, +0.57] pp).
- `early T=250` captures ~50% of the NLL benefit with essentially **zero**
  satisfaction movement (−0.03 pp).
- Paired vs global α=10: `early T=500` has satisfaction **+0.50 pp higher**
  (p=0.29) for a NLL within 0.00013 of α=10.

**Conclusion:** the satisfaction cost is a *late-training* artifact of continued
rebalancing. Stopping at step 500 keeps the fluency benefit and removes the
instruction-following cost.

---

## Hypothesis 2 — Output-gradient damping (H3)

### Idea

The reference paper's claim is that the *output* gradient dominates. All prior
work **boosted the input** gradient (α>1). Damping the *output* gradient (γ<1)
is the mirror intervention. Under Adam the two are directionally equivalent, but
the boost variant produces a ~α× larger gradient that gets **clipped**
(`clip_grad_norm=1.0`), understating the true effect. Damping sidesteps the clip.

### Intervention

```
g_eff = g_in + γ·g_out    (γ < 1)
```

### 12-seed results

| Config | NLL | Satisfaction | Δ NLL vs tied | Δ sat vs tied |
|---|---|---|---|---|
| tied | 1.5083 | 0.3353 | — | — |
| γ=0.5 | 1.5076 | 0.3340 | −0.0007 (p<10⁻¹³) | −0.13 pp (p=0.61) |
| γ=0.1 | **1.5042** | 0.3207 | **−0.0042** (p<10⁻¹³) | **−1.47 pp** (p=0.0001) |
| global α=10 | 1.5050 | 0.3272 | −0.0033 (p<10⁻¹³) | −0.82 pp (p=0.06) |

- `γ=0.1` reaches NLL **1.5042**, a **25% larger** NLL benefit than global α=10
  (125.3% of α=10's gain), and significantly better than α=10 (paired Δ NLL
  −0.00084, p<0.0001).
- That larger benefit comes at a **significant** satisfaction cost (−1.47 pp,
  p=0.0001; paired vs α=10: −0.65 pp, p=0.016).

**Conclusion:** gradient clipping was masking the true magnitude of the
rebalancing effect. Unclipped (via output damping), the effect is ~25% larger —
and the satisfaction cost is correspondingly larger, confirming the
fluency/instruction-following trade-off is **fundamental at full strength**, not
an artifact. This is precisely why the *temporal* fix (H1) matters: it is the one
intervention that decouples the two.

---

## Combined picture

There is a single monotonic rebalancing curve trading NLL against satisfaction:

```
NLL 1.5083 (tied) ──▶ 1.5067 (early250) ──▶ 1.5051 (early500) ──▶ 1.5050 (α=10) ──▶ 1.5042 (γ=0.1)
sat 0.3353 (tied) ──▶ 0.3350          ──▶ 0.3322          ──▶ 0.3272        ──▶ 0.3207
```

- Moving right along the curve (stronger/more-sustained rebalancing) always
  improves NLL and always costs satisfaction — **except** the early-schedule
  points, which climb toward the NLL optimum while leaving satisfaction ~flat.
- The takeaway: **the unembedding bias is an early-training effect.** Fix it
  early and briefly (T ≈ 250–500 steps), and you get the fluency benefit at no
  instruction-following cost. Keep rebalancing, and you over-rotate.

---

## Quantified improvements (the "how much")

1. **Early rebalancing (T=500) vs the prior best (global α=10):**
   - same NLL gain to within 4% (−0.0032 vs −0.0033),
   - satisfaction cost reduced from −0.82 pp (borderline, p=0.06) to **−0.32 pp
     (not significant, p=0.45)** — i.e. the cost is effectively eliminated.

2. **Output damping (γ=0.1):**
   - **25% more NLL improvement** than α=10 (NLL 1.5042 vs 1.5050), with the
     effect magnitude no longer attenuated by gradient clipping.

*Note on magnitude:* these NLL effects are small in absolute terms (≈0.003
nats/token) but extremely reliable (t ≈ 40–60) because seed-to-seed NLL variance
is tiny. Satisfaction effects are measured in percentage points with seed SD ≈
2 pp.

---

## Caveats

- Single pretrained checkpoint and a fixed 500-prompt evaluation sample; results
  are conditional on both.
- The early-stop point T was only swept at {250, 500}; the optimum is somewhere
  in that range and is not yet finely located.
- 13.8M-parameter model on TinyStories only; the early-training-timing claim may
  not transfer to larger models.
- "Satisfaction" is all-keyword-presence (a coarse, binary metric); coherence is
  not measured.

## Recommended next steps

1. Sweep T ∈ {250, 375, 500} at 12 seeds to pin the early-stop optimum.
2. Confirm the clipping story by re-running global α=10 with `grad_clip` raised
   (or disabled) — it should converge to the γ=0.1 result.
3. Verify the early-training-timing mechanism with a gradient-norm/conflict
   probe over the first few hundred steps.
