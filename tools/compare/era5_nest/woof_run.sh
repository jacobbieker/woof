#!/usr/bin/env bash
# One `woof go` of a matched config, under the shared-card lock, with the
# per-process device memory sampled beside it.
#
#   woof_run.sh CONFIG OUTDIR [extra woof go arguments...]
#
# Environment:
#   WOOF          the woof command (default: "python -m woof")
#   GPU_LOCK      lock file (default /tmp/woof-gpu-test.lock)
#   CUPY_GPU_MEMORY_LIMIT  device-memory cap in bytes; unset = no cap (the
#                 clean-card timings), the shared-card runs set 6442450944
set -euo pipefail
config=${1:?usage: woof_run.sh CONFIG OUTDIR [args...]}
outdir=${2:?usage: woof_run.sh CONFIG OUTDIR [args...]}
shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCK=${GPU_LOCK:-/tmp/woof-gpu-test.lock}
WOOF=${WOOF:-python -m woof}
mkdir -p "$outdir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log=$outdir/go-$stamp.log

# shellcheck disable=SC2086
flock "$LOCK" bash -c '
    log=$1; sampler=$2; shift 2
    start=$(date +%s)
    "$@" > "$log" 2>&1 &
    pid=$!
    "$sampler" --pid $pid --out "${log%.log}.gpumem.tsv" &
    sampled=$!
    status=0
    wait $pid || status=$?
    wait $sampled 2>/dev/null || true
    end=$(date +%s)
    echo "woof_run: exit $status, $((end - start)) s wall (lock held)" | tee -a "$log"
    exit $status' _ "$log" "$HERE/gpu_memory_sampler.sh" \
    $WOOF go "$config" --outdir "$outdir" "$@"
echo "$log"
