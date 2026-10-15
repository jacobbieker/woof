#!/usr/bin/env bash
# Full-size woof vs Breeze timing for the FastEddy dry CBL -- for a CLEAN card.
#
# Every variant runs at every size, REPEATS times, interleaved (the variant
# order rotates each repeat so no model always runs first or last on a warm
# card).  Each process: build, WARMUP steps (JIT / NVRTC compilation and
# first-touch allocation, excluded), then REPS timed blocks of STEPS steps
# each bracketed by a device synchronize.  analyze.py takes the median over
# every timed block of every repeat.  No output is written inside the timed
# loop.
#
# Optional PROFILES=1 then runs one simulate pass per profile variant at
# PROFILE_SIZE for PROFILE_HOURS (default 256x256x128, 4 h: the FastEddy
# figure's time) and adds the profile comparison to the report.
#
# Environment (defaults in brackets):
#   PY        python with woof's runtime deps (cupy)            [python3]
#   JULIA     pinned julia 1.12.7      [$SCRATCH/julia-1.12.7/bin/julia]
#   JULIA_DEPOT_PATH                   [$SCRATCH/depot]
#   SCRATCH   [/home/jacob/wf-scratch/les-benchmark]
#   OUT       results directory        [$SCRATCH/out/full-<timestamp>]
#   SIZES     ["256x256x128 384x384x192"]
#   REPEATS [3]  WARMUP [20]  STEPS [100]  REPS [3]  SUBSTEPS [12]
#   VARIANTS  space-separated subset of the names below           [all]
#   LOCK      if set, every GPU process runs under `flock $LOCK`
#   PROFILES [0]  PROFILE_SIZE [256x256x128]  PROFILE_HOURS [4]
#   PROFILE_VARIANTS ["woof_km3 woof_km2 breeze_anelastic_weno5_ci
#                      breeze_se_matched breeze_anelastic_matched"]
#
# Expected cost on an RTX A6000 (ESTIMATE, not measured at these sizes):
# each Breeze process spends a few minutes compiling before its timed
# steps; the default matrix is 7 variants x 2 sizes x 3 repeats = 42
# processes, about 1.5-2.5 h wall.  PROFILES=1 adds roughly 1-3 h.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SCRATCH=${SCRATCH:-/home/jacob/wf-scratch/les-benchmark}
PY=${PY:-python3}
JULIA=${JULIA:-$SCRATCH/julia-1.12.7/bin/julia}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$SCRATCH/depot}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
OUT=${OUT:-$SCRATCH/out/full-$(date -u +%Y%m%dT%H%M%SZ)}
SIZES=${SIZES:-"256x256x128 384x384x192"}
REPEATS=${REPEATS:-3}
WARMUP=${WARMUP:-20}
STEPS=${STEPS:-100}
REPS=${REPS:-3}
SUBSTEPS=${SUBSTEPS:-12}
PROFILES=${PROFILES:-0}
PROFILE_SIZE=${PROFILE_SIZE:-256x256x128}
PROFILE_HOURS=${PROFILE_HOURS:-4}
PROFILE_VARIANTS=${PROFILE_VARIANTS:-"woof_km3 woof_km2 breeze_anelastic_weno5_ci breeze_se_matched breeze_anelastic_matched"}
mkdir -p "$OUT/timing" "$OUT/logs" "$OUT/profiles" "$TMPDIR"

WOOF="$PY $HERE/run_woof.py"
BREEZE="$JULIA --project=$HERE/julia $HERE/run_breeze.jl"
declare -A CMD=()
# shellcheck source=variants.sh
. "$HERE/variants.sh"
ORDER=(woof_km3 breeze_anelastic_weno5_ci breeze_se_matched woof_km2 breeze_se_weno5_ci breeze_anelastic_matched breeze_se_weno5_smag)
if [[ -n "${VARIANTS:-}" ]]; then read -r -a ORDER <<< "$VARIANTS"; fi
for v in "${ORDER[@]}"; do [[ -n "${CMD[$v]:-}" ]] || { echo "unknown variant $v" >&2; exit 2; }; done

gpu() { if [[ -n "${LOCK:-}" ]]; then flock "$LOCK" "$@"; else "$@"; fi; }

{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.max.sm --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
  echo "sizes: $SIZES repeats: $REPEATS warmup: $WARMUP steps: $STEPS reps: $REPS substeps: $SUBSTEPS"
  echo "variants: ${ORDER[*]}"
} | tee "$OUT/environment.txt"

n=${#ORDER[@]}
for ((r = 1; r <= REPEATS; r++)); do
  for size in $SIZES; do
    for ((i = 0; i < n; i++)); do
      v=${ORDER[$(((i + r - 1) % n))]}
      stem="$OUT/timing/${v}_${size}_r${r}"
      echo "[$(date -u +%T)] repeat $r size $size $v"
      # shellcheck disable=SC2086
      if ! gpu ${CMD[$v]} --mode timing --size "$size" --warmup "$WARMUP" \
          --steps "$STEPS" --reps "$REPS" --out "$stem" \
          > "$OUT/logs/${v}_${size}_r${r}.log" 2>&1; then
        echo "  FAILED (see $OUT/logs/${v}_${size}_r${r}.log)" >&2
      fi
      tail -n 1 "$OUT/logs/${v}_${size}_r${r}.log" || true
    done
  done
done

profile_stems=()
if [[ "$PROFILES" == "1" ]]; then
  for v in $PROFILE_VARIANTS; do
    stem="$OUT/profiles/${v}_${PROFILE_SIZE}"
    echo "[$(date -u +%T)] profile $v $PROFILE_SIZE ${PROFILE_HOURS} h"
    # shellcheck disable=SC2086
    if gpu ${CMD[$v]} --mode simulate --size "$PROFILE_SIZE" \
        --hours "$PROFILE_HOURS" --sample-every 300 --out "$stem" \
        > "$OUT/logs/profile_${v}.log" 2>&1; then
      profile_stems+=("$stem")
    else
      echo "  FAILED (see $OUT/logs/profile_${v}.log)" >&2
    fi
  done
fi

"$PY" "$HERE/analyze.py" --timing "$OUT/timing" \
  ${profile_stems[@]:+--profiles "${profile_stems[@]}"} --out "$OUT/report" \
  || echo "analysis failed" >&2
echo "report: $OUT/report/summary.md"
