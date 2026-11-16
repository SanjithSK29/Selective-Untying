# Reproducibility checklist

Use this checklist before sharing a new run, analysis, or result update. It
helps keep new evidence comparable with the committed experiments.

## Environment

- [ ] Record the Python, PyTorch, NumPy, tokenizers, datasets, and
      `huggingface_hub` versions.
- [ ] Record whether the run used CPU or GPU, including the CUDA build when
      applicable.
- [ ] Install dependencies from the repository requirements files.
- [ ] Run the unit-test suite before starting an expensive run.

## Inputs

- [ ] Record the checkpoint path and SHA-256 checksum.
- [ ] Record dataset revisions and the generated source manifest.
- [ ] Record tokenizer provenance and checksum.
- [ ] Record the calibration prefix, fine-tuning prefix, evaluation sample,
      filtering rules, and prompt count.
- [ ] Check for exact overlap when a new evaluation or data source is used.

## Run settings

- [ ] Record method names, selected-token budget, learning rate, batch size,
      update count, clipping settings, and all non-default arguments.
- [ ] Record every random seed and whether methods share minibatch order.
- [ ] Use a new output and artifact directory when settings change.
- [ ] Preserve per-seed records, generated responses, and per-example metrics
      needed for later audits.

## Analysis

- [ ] Pair method comparisons by seed.
- [ ] Report the number of seeds and the uncertainty calculation.
- [ ] Identify exploratory, post-hoc, or subset analyses explicitly.
- [ ] Separate response NLL, keyword recall, and keyword satisfaction from
      claims about human-rated coherence or fluency.
- [ ] Rebuild reports from the saved records and inspect the generated files.

## Before publishing

- [ ] Run `python -m unittest discover -v`.
- [ ] Run the relevant report and overlap commands.
- [ ] Review `git diff --check`.
- [ ] Confirm that datasets, checkpoints, credentials, and local run outputs
      are not included in the change.
- [ ] State any unrun checks or environment differences in the pull request.
