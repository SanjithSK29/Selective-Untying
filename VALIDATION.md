# Verification record

Updated November 10, 2026. Historical result files are preserved.

## Source and experimental records

The main experiment is the author's committed results_v2.json at 05ec9844:
48 runs, four methods, seeds 0–11, 500 prompts each. All main seeds 0–2 exactly
reproduce the original satisfaction/NLL values. All missing-word counts agree
with reported satisfaction; reconstructed prompts also match the recall records.
The historical GPU environment (Python 3.13, torch 2.14.1+cu130, NumPy 2.5.2,
tokenizers 0.23.3, tqdm 4.67.3, datasets 5.1.0, huggingface_hub 1.33.0) was supplied
by the author. requirements.txt preserves those versions.

## Completed checks

- Eleven local unit tests covering strict checkpoint loading and rejection of
  corruption; zero-offset/full-head initial logit equivalence; input/output gradient
  decomposition against independently untied weights; response masking and per-example
  NLL; retained missing-word/recall outputs; alpha=1 equivalence; CPU training;
  resume provenance guards; CSV record preservation; paired interval mathematics;
  main-result coverage/reproduction; overlap/subset integrity.
- Full 13.8M-model synthetic CPU smoke runs: one pretraining step with validation
  and checkpoint save; tied/full/conflict SFT; alpha=1 and alpha=2 SFT. Saved
  alpha=1 and tied weights matched exactly. These are debugging checks, not research
  experiments. The full-model smoke runs preceded the final metric/provenance merge;
  the final unit suite checks the merged behavior.
- Both main and exploratory report generators completed.
- Python compilation, relative README link checks and git diff whitespace checks.
- Author-supplied original checkpoint: strict model loading and finite forward pass,
  step 28,503, validation loss 1.4663459479808807. Received tokenizer configuration
  is structurally identical to the repository copy. Checksums are in reports/.
- The full official pretraining split (2,119,719 rows), the first 20,000 qualifying
  SFT examples and 10M-token calibration prefix were scanned. There were 129/500
  normalized exact pretraining matches, 0 SFT matches and 0 calibration matches.
  All 500 evaluation stories were unique. The 371-prompt subset analysis was rebuilt
  from the existing per-prompt lexical records without changing full-set results.

## Local verification environment

Python 3.14.6, torch 2.14.1+cpu, NumPy 2.5.3, tokenizers 0.23.3,
tqdm 4.68.2, datasets 5.1.0, huggingface_hub 1.33.0, scipy 1.18.1,
matplotlib 3.11.2. These differ slightly from the author's historical GPU environment.
No claim is made that the latter was reinstalled or rerun locally in full.

## Limits

- Final hardened code has not undergone a complete 48-run GPU reproduction.
- Automated GitHub checks must run after publication; do not call them passed
  until the remote workflow reports success.
- Near-duplicate/semantic overlap was not tested. Original data revision hashes
  were not recorded during the experiments; the audit pins the official sources
  and checks alignment against the saved missing-word/recall records.
- No new independent pretrained checkpoints or random-selected token sets were run.
- Pretraining resume preserves data/RNG/optimizer state but has not been tested
  across an actual interrupted GPU training session.
- No human coherence/integration scores are available. Fluency claims were removed.
- Subset response NLL cannot be reconstructed from the historical aggregate records.

Run checks: python -m unittest discover -v
Rebuild reports: see README.md.
