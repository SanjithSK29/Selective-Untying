# Gradient-Guided Selective Untying in Small Language Models

Course project for UE24AM342AA1 (Advanced Foundations in Machine Learning), PES University.
Team **Commit and Pray**: Sanjith Suresh Kumar, Shriyadita Mohanty, Sindhu K S, Greeshma G.

## Question

Small language models usually *tie* the input and output embeddings. Recent work (Lopardo et al., Findings of ACL 2026, arXiv 2603.26663) shows the shared embedding is dominated early in training by the output-side gradient. We ask: **if we give only the most gradient-conflicted tokens (400 of 8,000) their own output-side offset Δᵢ and keep the rest tied, does that improve instruction following more than simple selectors (frequency, frequency-matched random, gradient norm) at the same parameter budget?**

Pre-registered success criterion: conflict selection counts as positive only if mean all-keyword satisfaction is at least 2 pp above both the frequency and frequency-matched-random baselines over 3 paired seeds, with no mean NLL degradation versus tied. Otherwise it is reported as null/mixed.

## Result (summary)

Selective untying is a **null/mixed result**: the criterion is not met. Full details are in [`findings.txt`](findings.txt).

| Method | All-keyword satisfaction (mean of 3 seeds) | Response NLL |
|---|---|---|
| Frequency-matched random | 0.347 | 1.5084 |
| **Conflict-selected (ours)** | 0.345 | 1.5088 |
| Tied (baseline) | 0.344 | 1.5074 |
| Frequency | 0.341 | 1.5071 |
| Grad-norm | 0.340 | 1.5065 |
| Full untying | 0.331 | 1.5048 |

- Conflict − frequency: +0.47 pp; conflict − freq-matched random: −0.20 pp (need ≥ +2 pp). Seed-to-seed spread (~4 pp) is much larger than the method differences.
- The conflict score is a genuinely different selector from frequency (Spearman ρ = 0.11, Jaccard overlap of selected sets 0.022). It picks character names and concrete nouns, not function words.
- Input-gradient scaling (the fix proposed in the reference paper), swept over α ∈ {1, 2, 5, 10}, gives a monotonic trade-off: NLL improves (1.5074 → 1.5040) while keyword satisfaction falls (0.344 → 0.328).

## Setup

- Model: 6-layer pre-LN decoder-only Transformer, d_model 384, 6 heads, GELU FFN 1536, learned positions, 8K byte-level BPE, context 256. 13.8M parameters (tied), 16.9M fully untied.
- Pretraining: TinyStories, one pass, ~470M tokens, 28,503 steps. Validation loss 1.466 (perplexity 4.33).
- SFT / evaluation: TinyStoriesInstruct converted to keyword-constrained story prompts; 20,000 SFT examples, 1,000 steps, lr 1e-4; 500 evaluation prompts.
- Hardware: one NVIDIA RTX 4060 Laptop GPU (8 GB), bf16 mixed precision.

## Datasets

| Dataset | Used for | Link |
|---|---|---|
| TinyStories | Pretraining | https://huggingface.co/datasets/roneneldan/TinyStories |
| TinyStoriesInstruct | SFT and gradient-conflict calibration | https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct |

Both are public. Run `python download_data.py` to fetch them into `dataset/`.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
```

Run the pipeline in this order (all scripts use the defaults below; `--help` lists every option):

| Step | Command | Output | Approx. time (RTX 4060) |
|---|---|---|---|
| 1. Pretrain | `python train.py` | `checkpoints/latest.pt`, `tokenizer.json` | ~78 min (+ ~8 min one-time tokenization) |
| 2. Gradient-conflict calibration | `python calibrate.py` | `calibration.json` | see `calibrate.py` |
| 3. Conflict-signal analysis | `python analyze.py` | printed report | seconds |
| 4. SFT: 6 methods × 3 seeds | `python sft.py` | `results.json` | ~200 s per run (18 runs) |
| 5. Input-gradient scaling | `python sft_gradscale.py` | `results_gradscale.json` | ~200 s per run (α = 2, 5, 10; α = 1 is the tied run from step 4) |

Quick smoke tests: `python train.py --max_steps 2000`, `python sft.py --methods tied conflict --seeds 0`.

**Skip pretraining:** download the pretrained checkpoint and place it at `checkpoints/latest.pt`:
https://huggingface.co/ishrya/selective-untying/resolve/main/latest.pt (166 MB; includes optimizer state)

```bash
mkdir -p checkpoints && wget -O checkpoints/latest.pt https://huggingface.co/ishrya/selective-untying/resolve/main/latest.pt
```
`calibration.json` (the 400 selected tokens) is already in this repo, so steps 4 and 5 can run directly from the checkpoint.

## Repository layout

| File | Purpose |
|---|---|
| `transformer_architecture.py` | Model with tied / fully untied / selective-offset output modes |
| `train.py` | Tokenizer, data prep, pretraining loop |
| `calibrate.py` | Two-pass gradient decomposition; conflict score = −cos(g_in, g_out) per token |
| `analyze.py` | Conflict vs frequency / gradient norm, selection overlap, top tokens |
| `sft.py` | SFT and evaluation for the 6 methods |
| `sft_gradscale.py` | SFT with input-gradient scaling |
| `common.py` | Shared helpers (tokenizer, record parsing, checkpoint loading) |
| `findings.txt` | Full write-up of results and limitations |
| `*.log`, `results*.json`, `calibration.json` | Logs and raw outputs from our runs |

## Limitations

Single pretrained checkpoint; 3 seeds (underpowered for a 2 pp effect given ~4 pp seed spread); 13.8M-parameter model on TinyStories only; keyword satisfaction is a coarse binary metric; blinded human ratings were not run. See `findings.txt` section 9.

## References

[3] Lopardo et al., Findings of ACL 2026, arXiv 2603.26663. Eldan and Li, "TinyStories", arXiv 2305.07759.
