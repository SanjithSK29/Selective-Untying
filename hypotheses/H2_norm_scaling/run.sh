#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec /home/sanjith/miniconda3/bin/python hypotheses/screen.py \
  --configs norm_scaling:5 norm_scaling:10 \
  --seeds 0 1 2 3 \
  --out hypotheses/H2_norm_scaling/results.json
