#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec /home/sanjith/miniconda3/bin/python hypotheses/screen.py \
  --configs inverse_conflict:5 inverse_conflict:10 \
  --seeds 0 1 2 3 \
  --out hypotheses/H1_inverse_conflict/results.json
