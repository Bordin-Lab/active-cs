#!/usr/bin/env bash
set -euo pipefail

CASES=${1:-ibi_cases.tsv}
OUTROOT=${OUTROOT:-ibi_runs}
PYPRESSO=${PYPRESSO:-pypresso}
N=${N:-2000}
DIM=${DIM:-3}
MAX_ITER=${MAX_ITER:-8}
TOL=${TOL:-1e-3}
ALPHA0=${ALPHA0:-0.08}

mkdir -p "$OUTROOT"

tail -n +2 "$CASES" | while IFS=$'\t' read -r T rho v pot rdf out_name; do
  echo "=== IBI case: T=$T rho=$rho v=$v ==="
  python run_ibi_optimized.py \
    --pypresso "$PYPRESSO" \
    --initial-potential "$pot" \
    --target-rdf "$rdf" \
    --N "$N" --rho "$rho" --T "$T" --dim "$DIM" \
    --out "$OUTROOT/$out_name" \
    --max-iter "$MAX_ITER" --tol "$TOL" --alpha0 "$ALPHA0" \
    --log --resume
 done
