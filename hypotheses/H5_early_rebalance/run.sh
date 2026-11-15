#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec /home/sanjith/miniconda3/bin/python hypotheses/screen.py \
  --configs early_rebalance:10@250 early_rebalance:10@500 \
  --seeds 0 1 2 3 \
  --out hypotheses/H5_early_rebalance/results.json
