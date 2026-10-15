#!/bin/bash
# Small end-to-end runs of both models on the shared card, in accuracy mode
# (warm-up + timed block, then integrate to the end time and dump fields).
# All ClimaAtmos runs share one Julia session (one compile, one GPU-lock
# hold); each woof run takes the lock separately.  env.sh caps GPU memory.
#
#   bash run_e2e_small.sh                 # Schaer (SCHAR_HOURS) + density currents
#   SCHAR_HOURS=6 bash run_e2e_small.sh
#   SKIP_CA=1 bash run_e2e_small.sh       # woof side only
#
# The Schaer runs are shortened (SCHAR_HOURS, default 2 h) relative to the
# 24 h ClimaAtmos config (the full length is in run_timing.sh ACCURACY=1);
# the density currents run the full 900 s.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCHAR_HOURS="${SCHAR_HOURS:-2}"
case "$SCHAR_HOURS" in
    ''|*[!0-9]*) echo "SCHAR_HOURS must be a whole number of hours"; exit 2 ;;
esac
SKIP_CA="${SKIP_CA:-0}"
SKIP_WOOF="${SKIP_WOOF:-0}"

if [ "$SKIP_CA" != "1" ]; then
    ca_batch_begin
    ca_add e2e_ca_schar --config "$CA_CFG/plane_schar_mountain_float32_test.yml" \
        --set "t_end=${SCHAR_HOURS}hours" --warm-steps 20 --timed-steps 200
    ca_add e2e_ca_dc_stock --config "$CA_CFG/plane_density_current_test.yml" \
        --warm-steps 20 --timed-steps 200
    # Must follow every stock density-current run (in-session IC redefinition).
    ca_add e2e_ca_dc_straka --config "$CA_CFG/plane_density_current_test.yml" \
        --straka-ic --warm-steps 20 --timed-steps 200
    ca_batch_run e2e
fi

if [ "$SKIP_WOOF" != "1" ]; then
    woof_run e2e_woof_schar --case schar --run-seconds $((SCHAR_HOURS * 3600)) \
        --warm-steps 20 --timed-steps 200
    # WRF fill_w start (hill2d's), for the start-up transient comparison.
    woof_run e2e_woof_schar_wrffill --case schar --w-init wrf-fill \
        --run-seconds $((SCHAR_HOURS * 3600)) --warm-steps 20 --timed-steps 200
    woof_run e2e_woof_dc_straka --case dc --ic straka --warm-steps 20 --timed-steps 200
    woof_run e2e_woof_dc_clima --case dc --ic clima --warm-steps 20 --timed-steps 200
    # woof's gated Straka grid (dx = dz = 100 m, dt = 0.5 s) for reference.
    woof_run e2e_woof_dc_straka_100m --case dc --ic straka --nx 512 --nz 64 \
        --dt 0.5 --warm-steps 20 --timed-steps 200
fi
summarize
