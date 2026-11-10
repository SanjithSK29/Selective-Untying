# Gradient-Guided Selective Untying in Small Language Models

Does giving only the most gradient-conflicted tokens their own output embedding help a small tied-embedding language model follow instructions? This repo contains the full pipeline (pretraining, gradient-conflict calibration, fine-tuning, evaluation) and the results. **Short answer: no detectable effect at this scale (a null result).**

## Contents

1. [Overview](#overview)
2. [Results at a glance](#results-at-a-glance)
3. [Quick start](#quick-start)
4. [Full pipeline](#full-pipeline)
5. [Experimental setup](#experimental-setup)
6. [Repository layout](#repository-layout)
7. [Limitations](#limitations)
8. [References](#references)

## Overview

Small language models usually *tie* the input and output embeddings, so one table serves both reading tokens and predicting them. Lopardo et al. (Findings of ACL 2026, arXiv 2603.26663) show the shared embedding is dominated early in training by the output-side gradient.

**Hypothesis.** For a small set of tokens whose input-side and output-side gradients conflict most, add an output-only offset Δᵢ (initialised to 0) and keep everything else tied. This is cheaper and more targeted than fully untying the embeddings.

**Pre-registered success criterion.** Conflict selection counts as positive only if mean all-keyword satisfaction is at least 2 percentage points above *both* the frequency and frequency-matched-random baselines over 3 paired seeds, with no mean degradation in response NLL versus the tied model. Otherwise it is reported as null/mixed.

## Results at a glance

**Selective untying (400 untied tokens, 3 seeds each)**

| Method | All-keyword satisfaction | Response NLL |
|---|---|---|
| Frequency-matched random | 0.347 | 1.5084 |
| **Conflict-selected (ours)** | 0.345 | 1.5088 |
| Tied (baseline) | 0.344 | 1.5074 |
| Frequency | 0.341 | 1.5071 |
| Grad-norm | 0.340 | 1.5065 |
| Full untying | 0.331 | 1.5048 |

- Conflict − frequency = +0.47 pp; conflict − frequency-matched random = −0.20 pp. The bar was +2 pp, so the criterion is **not met**.
- Seed-to-seed spread (~4 pp) is much larger than the differences between methods.
- The conflict score is a genuinely different selector from frequency: Spearman ρ = 0.11, and the selected sets overlap only 2.2% (Jaccard). It picks character names and concrete nouns rather than function words.

**Input-gradient scaling** (the fix proposed in the reference paper; tied model, 3 seeds each)

| α | Response NLL | All-keyword satisfaction |
|---|---|---|
| 1 (tied) | 1.5074 | 0.344 |
| 2 | 1.5068 | 0.342 |
| 5 | 1.5054 | 0.335 |
| 10 | 1.5040 | 0.328 |

Next-token prediction improves monotonically while instruction following degrades monotonically: a fluency / instruction-following trade-off. Full write-up: [`findings.txt`](findings.txt).

## Quick start

Requires Python 3.13 and an NVIDIA GPU (developed on an RTX 4060 Laptop, 8 GB, bf16).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. SFT / calibration data (~2.7 GB)
python download_data.py --only instruct

# 2. Pretrained checkpoint (166 MB, includes optimizer state)
mkdir -p checkpoints
wget -O checkpoints/latest.pt https://huggingface.co/ishrya/selective-untying/resolve/main/latest.pt

# 3. Smoke test (about a minute)
python sft.py --methods tied conflict --seeds 0 --max_steps 50 --eval_examples 50 --out smoke.json
```

`tokenizer.json` and `calibration.json` (the 400 selected tokens) are already in this repo, so you can skip pretraining and calibration and go straight to fine-tuning.

## Full pipeline

Run the steps in order. Every script has sensible defaults; `--help` lists all options.

| Step | Command | Output | Approx. time (RTX 4060) |
|---|---|---|---|
| 0. Data | `python download_data.py` | `dataset/` | download-bound |
| 1. Pretrain | `python train.py` | `checkpoints/latest.pt`, `tokenizer.json` | ~78 min (+ ~8 min one-time tokenization) |
| 2. Calibrate | `python calibrate.py` | `calibration.json` | see `calibrate.py` |
| 3. Analyze conflict signal | `python analyze.py` | printed report | seconds |
| 4. SFT, 6 methods × 3 seeds | `python sft.py` | `results.json` | ~200 s per run, 18 runs |
| 5. Input-gradient scaling | `python sft_gradscale.py` | `results_gradscale.json` | ~200 s per run, α ∈ {2, 5, 10} (α = 1 is the tied run from step 4) |

Methods compared in step 4: `tied`, `full` (full untying), `conflict`, `frequency`, `freq_random` (frequency-matched random), `grad_norm`. All selective methods add the same 400 output offsets and differ only in *which* tokens are chosen. Runs are saved incrementally and skipped if already finished, so interrupted runs can be resumed.

## Experimental setup

| Item | Value |
|---|---|
| Model | 6-layer pre-LN decoder-only Transformer, d_model 384, 6 heads, GELU FFN 1536, learned positions |
| Vocabulary / context | 8,000 byte-level BPE tokens / 256 |
| Parameters | 13.8M tied; 16.9M fully untied |
| Pretraining | TinyStories, one pass, ~470M tokens, 28,503 steps; validation loss 1.466 (perplexity 4.33) |
| Conflict score | −cos(g_in, g_out) per token, from a two-pass gradient decomposition on 10M calibration tokens |
| Token eligibility | frequency ≥ 20, non-special; 6,341 eligible, top 400 selected |
| SFT | TinyStoriesInstruct as keyword-constrained story prompts; 20,000 examples, 1,000 steps, lr 1e-4 |
| Evaluation | All-keyword satisfaction (case-insensitive whole-word match in generated stories) and response NLL on 500 held-out prompts |

**Datasets** (public): [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for pretraining, [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) for SFT and calibration.

## Repository layout

| Path | Purpose |
|---|---|
| `transformer_architecture.py` | Model with tied / fully untied / selective-offset output modes |
| `train.py` | Tokenizer, data preparation, pretraining loop |
| `calibrate.py` | Gradient-conflict calibration |
| `analyze.py` | Conflict vs frequency / gradient norm, selection overlap, top tokens |
| `sft.py` | SFT and evaluation for the six methods |
| `sft_gradscale.py` | SFT with input-gradient scaling |
| `common.py` | Shared helpers (tokenizer, record parsing, checkpoint loading) |
| `download_data.py` | Dataset download |
| `tokenizer.json`, `calibration.json` | Trained tokenizer and calibration output |
| `results.json`, `results_gradscale.json` | Raw per-run results |
| `*.log` | Training, SFT and download logs |
| `findings.txt` | Full findings and limitations |

## Limitations

- One pretrained checkpoint; findings are conditional on it.
- Three seeds are underpowered for a 2 pp effect given ~4 pp seed spread.
- Small scale: a 13.8M-parameter model on TinyStories only; results may not transfer to larger models or other tasks.
- Keyword satisfaction is a coarse binary metric.
- Blinded human ratings of coherence were not run.
- SFT hyperparameters were not tuned.

More detail in `findings.txt`, section 9.

## References

- Lopardo et al., Findings of ACL 2026, arXiv 2603.26663.
- Eldan and Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", arXiv 2305.07759.
