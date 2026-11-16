# Contributing

Thank you for helping improve Selective Untying. Contributions should make the
experiments easier to inspect, reproduce, or extend without weakening the
project's claim boundaries.

## Before opening a change

1. Read [the evaluation protocol](docs/EVALUATION.md) and [the validation
   record](VALIDATION.md).
2. Keep generated datasets, checkpoints, and `runs/` outputs out of commits.
3. For an experiment, record the checkpoint, dataset revisions, prompt sample,
   seeds, configuration, and output paths.
4. Prefer a focused pull request with one logical change.

## Local checks

From the repository root, create an environment and install the analysis
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

Run the same checks used by continuous integration:

```bash
python -m unittest discover -v
python report_results.py --output runs/ci-report
python report_results.py --results results.json --scaling results_gradscale.json --output runs/ci-exploratory
```

Do not claim a full GPU reproduction from the CPU checks. If a check cannot be
run locally, describe the limitation in the pull request.

## Experiment and analysis changes

- Preserve historical result files; add a new output path for changed settings.
- Pair methods by seed when comparing outcomes.
- Report the sample size, uncertainty calculation, and the exact evaluation
  subset.
- Distinguish exploratory follow-ups from the original experiment.
- Do not describe lower reference NLL as a direct human fluency measurement.
- Do not add private data, credentials, downloaded checkpoints, or large
  generated artifacts.

## Pull requests

The pull-request template is a checklist, not a substitute for results
documentation. Include the motivation, files changed, checks run, and any
reproducibility or claim-boundary implications. Keep unrelated formatting or
refactoring out of the change.
