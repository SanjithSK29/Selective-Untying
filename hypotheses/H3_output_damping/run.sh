#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec /home/sanjith/miniconda3/bin/python hypotheses/screen.py \
  --configs output_damping:0.5 output_damping:0.1 \
  --seeds 0 1 2 3 \
  --out hypotheses/H3_output_damping/results.json
