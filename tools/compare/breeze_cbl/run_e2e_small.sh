#!/usr/bin/env bash
# Small end-to-end check of the woof/Breeze CBL harness on a SHARED card.
#
# 128 x 128 x 64 (dx = 93.75 m, dz = 46.9 m), HOURS (default 2) simulated hours
# per profile variant, plus a short timing pass per variant so the
# throughput table is exercised.  Every GPU process holds LOCK and runs with
# memory capped (CuPy pool 6 GiB, CUDA.jl hard limit 6 GiB).  The timings
# printed here come from a shared card and only show that the harness works;
# do not quote them as performance figures.  REUSE=1 keeps an existing
# simulate output ($OUT/<variant>.json) instead of re-running it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SCRATCH=${SCRATCH:-/home/jacob/wf-scratch/les-benchmark}
PY=${PY:-python3}
JULIA=${JULIA:-$SCRATCH/julia-1.12.7/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$SCRATCH/depot}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export CUPY_GPU_MEMORY_LIMIT=${CUPY_GPU_MEMORY_LIMIT:-6442450944}
export JULIA_CUDA_HARD_MEMORY_LIMIT=${JULIA_CUDA_HARD_MEMORY_LIMIT:-6GiB}
LOCK=${LOCK:-/tmp/woof-gpu-test.lock}
OUT=${OUT:-$SCRATCH/out/small}
SIZE=${SIZE:-128x128x64}
HOURS=${HOURS:-2}
mkdir -p "$OUT/timing" "$OUT/logs" "$TMPDIR"

WOOF="$PY $HERE/run_woof.py"
BREEZE="$JULIA --project=$HERE/julia $HERE/run_breeze.jl"
SUBSTEPS=${SUBSTEPS:-12}
declare -A CMD=()
# shellcheck source=variants.sh
. "$HERE/variants.sh"
VARIANTS=${VARIANTS:-"woof_km3 woof_km2 breeze_anelastic_weno5_ci breeze_se_matched"}

stems=()
for v in $VARIANTS; do
  echo "[$(date -u +%T)] simulate $v $SIZE ${HOURS} h"
  # shellcheck disable=SC2086
  if [[ "${REUSE:-0}" == "1" && -f "$OUT/$v.json" ]]; then
    echo "  reusing $OUT/$v.json"
    stems+=("$OUT/$v")
  elif flock "$LOCK" ${CMD[$v]} --mode simulate --size "$SIZE" \
      --hours "$HOURS" --sample-every 300 --out "$OUT/$v" > "$OUT/logs/sim_$v.log" 2>&1; then
    stems+=("$OUT/$v")
  else
    echo "  FAILED (see $OUT/logs/sim_$v.log)" >&2
  fi
  echo "[$(date -u +%T)] timing $v $SIZE"
  # shellcheck disable=SC2086
  flock "$LOCK" ${CMD[$v]} --mode timing --size "$SIZE" --warmup 10 --steps 50 \
      --reps 2 --out "$OUT/timing/$v" > "$OUT/logs/timing_$v.log" 2>&1 \
      || echo "  timing FAILED (see $OUT/logs/timing_$v.log)" >&2
done

"$PY" "$HERE/analyze.py" --timing "$OUT/timing" --profiles "${stems[@]}" \
    --window-minutes 30 --out "$OUT/report"
