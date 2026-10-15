#!/usr/bin/env bash
# Run the Breeze MC3E timing driver (breeze_mc3e_timing.jl) in the
# environment breeze_setup.sh installed.
#
#   breeze_run.sh prefetch   CPU, 10x coarsened grid over the same box, no
#                            step: downloads ERA5 (CDS), ETOPO2022 and the
#                            CGLS albedo.  No GPU.
#   breeze_run.sh time       GPU run of BREEZE_HOURS (default 1) at full size,
#                            under the shared-card lock and memory cap
#                            (BREEZE_MEMORY_CAP, default 6GiB; "off" = none),
#                            optionally bounded by BREEZE_TIMEOUT seconds.
#
# Credentials: the CDS client reads $HOME/.cdsapirc.  This script points HOME
# at $BENCH_ROOT/home so the url/key pair the CDS needs can live there
# without touching the real home directory; put a two-line rc file
# (url: https://cds.climate.copernicus.eu/api / key: <token>) at
# $BENCH_ROOT/home/.cdsapirc, or set CDS_RC to one.
set -euo pipefail
mode=${1:?usage: breeze_run.sh prefetch|time}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=${BENCH_ROOT:-$HOME/wf-scratch/regional-benchmark}
JULIA=$BENCH_ROOT/julia/julia-1.12.6/bin/julia
RUN_DIR=${RUN_DIR:-$BENCH_ROOT/breeze-mc3e}
LOCK=${GPU_LOCK:-/tmp/woof-gpu-test.lock}
mkdir -p "$RUN_DIR" "$BENCH_ROOT/home" "$BENCH_ROOT/tmp"
if [ -n "${CDS_RC:-}" ]; then
    install -m 600 "$CDS_RC" "$BENCH_ROOT/home/.cdsapirc"
fi
[ -f "$BENCH_ROOT/home/.cdsapirc" ] || { echo "no CDS rc at $BENCH_ROOT/home/.cdsapirc" >&2; exit 2; }

export HOME=$BENCH_ROOT/home
export JULIA_DEPOT_PATH=$BENCH_ROOT/julia/depot
export TMPDIR=$BENCH_ROOT/tmp
export BREEZE_ERA5_DIR=$RUN_DIR/era5
cd "$RUN_DIR"

case $mode in
prefetch)
    BREEZE_ARCH=CPU BREEZE_COARSEN=10 BREEZE_SETUP_ONLY=1 \
    BREEZE_HOURS=${BREEZE_HOURS:-3} BREEZE_SUMMARY=$RUN_DIR/prefetch.json \
        "$JULIA" --project="$BENCH_ROOT/julia/env" "$HERE/breeze_mc3e_timing.jl"
    ;;
time)
    # BREEZE_MEMORY_CAP: CUDA.jl hard device-memory limit (default 6GiB, the
    # shared-card cap); "off" runs uncapped.
    cap=${BREEZE_MEMORY_CAP:-6GiB}
    if [ "$cap" = off ]; then
        unset JULIA_CUDA_HARD_MEMORY_LIMIT
    else
        export JULIA_CUDA_HARD_MEMORY_LIMIT=$cap
    fi
    export BREEZE_SUMMARY=${BREEZE_SUMMARY:-$RUN_DIR/timing-$(date -u +%Y%m%dT%H%M%SZ).json}
    # BREEZE_TIMEOUT: seconds the GPU session may hold the lock (0 = none).
    flock "$LOCK" bash -c '
        timeout --signal=TERM "${BREEZE_TIMEOUT:-0}" "$1" "$2" "$3" &
        pid=$!
        "$4" --pid $pid --out "${BREEZE_SUMMARY%.json}.gpumem.tsv" &
        sampler=$!
        wait $pid; status=$?
        wait $sampler 2>/dev/null || true
        exit $status' _ "$JULIA" --project="$BENCH_ROOT/julia/env" \
        "$HERE/breeze_mc3e_timing.jl" "$HERE/gpu_memory_sampler.sh"
    ;;
*)
    echo "unknown mode $mode" >&2; exit 2;;
esac
