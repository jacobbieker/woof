#!/usr/bin/env bash
# Full-size timings for the clean-card phase: both Breeze ERA5 domains in
# woof, at the published run lengths, plus the Breeze MC3E driver on the
# same card.
#
#   tools/compare/era5_nest/run_timing.sh [mc3e] [ar] [breeze]   (default: all)
#
# Network stages (ERA5 fetch, Breeze prefetch) run outside the GPU lock;
# every GPU stage takes the lock.  No device-memory cap is applied here --
# these are the clean-card numbers; set CUPY_GPU_MEMORY_LIMIT (bytes) /
# BREEZE_MEMORY_CAP (e.g. 6GiB) to impose one.
#
# Environment:
#   BENCH_ROOT  scratch root (default ~/wf-scratch/regional-benchmark)
#   WOOF        woof command (default "python -m woof", run from the repo root
#               with PYTHONPATH pointing at it)
#   PY          python for the harness scripts (default python)
#   VRAM_GIB    card the configs are sized for (default 48, RTX A6000)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=${BENCH_ROOT:-$HOME/wf-scratch/regional-benchmark}
PY=${PY:-python}
WOOF=${WOOF:-$PY -m woof}
VRAM_GIB=${VRAM_GIB:-48}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS=$BENCH_ROOT/results-$STAMP
mkdir -p "$RESULTS"
stages=${*:-mc3e ar breeze}
export WOOF

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version \
    --format=csv > "$RESULTS/card-before.csv"

# fetch_case CASE HOURS -> prints the config path
fetch_case() {
    local case=$1 hours=$2 dir=$BENCH_ROOT/$1-full config
    config=$("$PY" "$HERE/make_nest_configs.py" --case "$case" --out-dir "$dir" \
        --hours "$hours" --vram-gib "$VRAM_GIB" --no-restarts | tail -1)
    # The [fetch] table the wizard wrote is the request; woof fetch runs it.
    read -r cycle fhours area cadence out < <("$PY" - "$config" <<'EOF'
import sys, tomllib
f = tomllib.load(open(sys.argv[1], "rb"))["fetch"]
print(f["cycle"], f["hours"], f["area"], f["cadence"], f["out"])
EOF
)
    (cd "$dir" && TIMEFORMAT="fetch: %R s wall" && time \
        $WOOF fetch --source era5 --era5-provider arco --retrieve \
        --cycle "$cycle" --hours "$fhours" --area "$area" --cadence "$cadence" \
        --out "$out") > "$RESULTS/$case-fetch.log" 2>&1
    du -sb "$dir/$out" > "$RESULTS/$case-fetch-bytes.txt"
    echo "$config"
}

run_case() {
    local case=$1 hours=$2 config run status=0
    config=$(fetch_case "$case" "$hours")
    "$HERE/woof_run.sh" "$config" "$BENCH_ROOT/$case-full/go-$STAMP" \
        > "$RESULTS/$case-woof-run.out" 2>&1 || status=$?
    cp "$BENCH_ROOT/$case-full/go-$STAMP"/go-*.log "$BENCH_ROOT/$case-full/go-$STAMP"/go-*.gpumem.tsv \
        "$RESULTS/" 2>/dev/null || true
    [ "${status:-0}" -eq 0 ] || { echo "$case: woof go exited ${status}" >&2; return 1; }
    "$PY" "$HERE/nest_throughput.py" --json woof "$BENCH_ROOT/$case-full/go-$STAMP" \
        --case "$case" > "$RESULTS/$case-woof-throughput.json"
    "$PY" "$HERE/nest_throughput.py" woof "$BENCH_ROOT/$case-full/go-$STAMP" --case "$case"
    # ASOS scores; `woof go` verifies in the background, so run it to completion.
    run=$(ls -d "$BENCH_ROOT/$case-full/go-$STAMP"/run-* | tail -1)
    $WOOF verify-visuals "$run" --cycle "$(date -u -d "${CYCLE[$case]}" +%Y-%m-%dT%HZ)" \
        > "$RESULTS/$case-verify.log" 2>&1 || true
    "$PY" "$HERE/station_table.py" "$run" > "$RESULTS/$case-stations.md" || true
}

declare -A CYCLE=([mc3e]="2011-05-20 00:00" [ar]="2025-12-07 12:00")

failed=()
for stage in $stages; do
    case $stage in
    mc3e) run_case mc3e 3 || failed+=(mc3e) ;;
    ar)   run_case ar 72 || failed+=(ar) ;;
    breeze)
        if BREEZE_HOURS=3 "$HERE/breeze_run.sh" prefetch > "$RESULTS/breeze-prefetch.log" 2>&1 \
            && BREEZE_MEMORY_CAP=${BREEZE_MEMORY_CAP:-off} \
               BREEZE_HOURS=3 BREEZE_SUMMARY=$RESULTS/breeze-mc3e-timing.json \
               "$HERE/breeze_run.sh" time > "$RESULTS/breeze-mc3e.log" 2>&1; then
            "$PY" "$HERE/nest_throughput.py" breeze "$RESULTS/breeze-mc3e-timing.json" --case mc3e
        else
            failed+=(breeze)
        fi
        ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done

for case in mc3e ar; do
    "$PY" "$HERE/nest_throughput.py" published --case $case --scale-to rtx-a6000
done > "$RESULTS/published.md"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv > "$RESULTS/card-after.csv"
echo "results in $RESULTS"
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed stages: ${failed[*]}" | tee "$RESULTS/FAILED"
    exit 1
fi
