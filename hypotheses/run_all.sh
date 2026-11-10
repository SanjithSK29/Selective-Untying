#!/usr/bin/env bash
# Run all five hypothesis screenings sequentially (each ~27 min).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for h in H1_inverse_conflict H2_norm_scaling H3_output_damping H4_orthogonalize H5_early_rebalance; do
  echo "=== START $h $(date '+%F %T') ==="
  bash "hypotheses/$h/run.sh" || echo "!!! $h FAILED (rc=$?)"
  echo "=== END   $h $(date '+%F %T') ==="
done
echo "=== ALL SCREENING DONE $(date '+%F %T') ==="
