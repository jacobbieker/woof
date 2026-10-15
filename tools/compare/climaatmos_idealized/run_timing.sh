#!/bin/bash
# Throughput (and optionally full-length accuracy) runs for the clean-card
# phase.  Not run at these sizes by the author (shared card): see README.md
# and docs/comparisons/clima-numericalearth/idealized-benchmark.md.
#
#   RUNS_DIR=/path/to/clean-card-runs bash run_timing.sh
#   ACCURACY=1 bash run_timing.sh      # also 24 h Schaer + 900 s density currents
#   REPEATS=3 bash run_timing.sh       # repeat each timing run
#
# On a dedicated card set JULIA_CUDA_HARD_MEMORY_LIMIT=none and
# CUPY_GPU_MEMORY_LIMIT=0 (env.sh defaults cap both models at 6 GiB for the
# shared card).  Each timing run: warm-up block (JIT/kernel compilation,
# allocator) excluded, then TIMED steps timed between device
# synchronisations, no diagnostics or state output.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
REPEATS="${REPEATS:-1}"
WARM="${WARM:-50}"
TIMED="${TIMED:-1000}"
ACCURACY="${ACCURACY:-0}"
T="--mode timing --warm-steps $WARM --timed-steps $TIMED"
SCHAR="$CA_CFG/plane_schar_mountain_float32_test.yml"
DC="$CA_CFG/plane_density_current_test.yml"
BOX="$CA_CFG/box_density_current_test.yml"

if [ "${JULIA_CUDA_HARD_MEMORY_LIMIT}" = "none" ]; then
    unset JULIA_CUDA_HARD_MEMORY_LIMIT
fi
if [ "${CUPY_GPU_MEMORY_LIMIT}" = "0" ]; then
    unset CUPY_GPU_MEMORY_LIMIT
fi
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv | tee "$RUNS_DIR/gpu_state_before.csv"

# ---- ClimaAtmos: one Julia session for every run -------------------------
ca_batch_begin
for r in $(seq 1 "$REPEATS"); do
    # 1. Schaer, the config's grid: 100 x 100 elements (300 unique x nodes,
    #    100 levels), dt 0.7 s.
    ca_add "t${r}_ca_schar" --config "$SCHAR" $T
    # 2. 4x horizontal resolution (1200 unique x nodes, dx = 83 m).
    ca_add "t${r}_ca_schar_x4" --config "$SCHAR" --set x_elem=400 \
        --set dt=0.2secs $T
    # 3. 2-D density current, the config's grid (240 x 90), dt 0.3 s.
    ca_add "t${r}_ca_dc" --config "$DC" $T
    # 4. 3-D density current box: 45^3 elements (135 x 135 x 45), dt 0.25 s.
    ca_add "t${r}_ca_box_dc" --config "$BOX" $T
    # 5. 4x the points (90 x 90 x 45 elements), dt 0.125 s.
    ca_add "t${r}_ca_box_dc_x4" --config "$BOX" --set x_elem=90 \
        --set y_elem=90 --set dt=0.125secs $T
done
if [ "$ACCURACY" = "1" ]; then
    ca_add acc_ca_schar_24h --config "$SCHAR" --warm-steps 50 --timed-steps 1000
    ca_add acc_ca_dc_stock --config "$DC" --warm-steps 50 --timed-steps 1000
    ca_add acc_ca_dc_straka --config "$DC" --straka-ic --warm-steps 50 \
        --timed-steps 1000
fi
ca_batch_run timing

# ---- woof -----------------------------------------------------------------
for r in $(seq 1 "$REPEATS"); do
    # Schaer at woof's own dt (2 s) and at ClimaAtmos's (0.7 s).
    woof_run "t${r}_woof_schar" --case schar $T
    woof_run "t${r}_woof_schar_dt0.7" --case schar --dt 0.7 $T
    woof_run "t${r}_woof_schar_x4" --case schar --nx 1200 --dt 0.5 $T
    woof_run "t${r}_woof_dc" --case dc --ic straka $T
    # 3-D: constant K = 75, y-uniform bubble (throughput only).
    woof_run "t${r}_woof_box_dc" --case dc --ic straka --nx 135 --ny 135 \
        --nz 45 --dt 1.0 $T
    woof_run "t${r}_woof_box_dc_x4" --case dc --ic straka --nx 270 --ny 270 \
        --nz 45 --dt 0.5 $T
done
if [ "$ACCURACY" = "1" ]; then
    woof_run acc_woof_schar_24h --case schar --run-seconds 86400 \
        --warm-steps 50 --timed-steps 1000
    woof_run acc_woof_dc_straka --case dc --ic straka --warm-steps 50 --timed-steps 500
    woof_run acc_woof_dc_clima --case dc --ic clima --warm-steps 50 --timed-steps 500
fi

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv | tee "$RUNS_DIR/gpu_state_after.csv"
summarize
