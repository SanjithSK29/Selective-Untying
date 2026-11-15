#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec /home/sanjith/miniconda3/bin/python hypotheses/screen.py \
  --configs orthogonalize:0.5 orthogonalize:1.0 \
  --seeds 0 1 2 3 \
  --out hypotheses/H4_orthogonalize/results.json
