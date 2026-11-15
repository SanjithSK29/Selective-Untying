#!/usr/bin/env bash
# 12-seed confirmatory runs for the two promising hypotheses.
# Resumes from the screening results (seeds 0-3 already done), adding seeds 4-11.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/sanjith/miniconda3/bin/python
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"

echo "=== CONFIRM H5 early_rebalance (12 seeds) $(date '+%F %T') ==="
"$PY" hypotheses/screen.py \
  --configs early_rebalance:10@250 early_rebalance:10@500 \
  --seeds $SEEDS --out hypotheses/H5_early_rebalance/results.json

echo "=== CONFIRM H3 output_damping (12 seeds) $(date '+%F %T') ==="
"$PY" hypotheses/screen.py \
  --configs output_damping:0.5 output_damping:0.1 \
  --seeds $SEEDS --out hypotheses/H3_output_damping/results.json

echo "=== ALL CONFIRMATORY DONE $(date '+%F %T') ==="
