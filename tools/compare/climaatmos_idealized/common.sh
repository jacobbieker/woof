# Shared helpers for run_e2e_small.sh / run_timing.sh (source, don't run).
# shellcheck shell=bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CA_CFG="$CA_REPO/config/model_configs"

# clear_run DIR: drop a previous run's outputs so a failed rerun cannot leave
# stale numbers for `summarize` to pick up.
clear_run() {
    rm -rf "$1/result.json" "$1/fields" "$1/fields.npz" "$1/ca_output"
    mkdir -p "$1"
}

# ca_run NAME [driver args...]: one ClimaAtmos run under the GPU lock.
ca_run() {
    local name=$1; shift
    local out="$RUNS_DIR/$name"
    clear_run "$out"
    echo "== ClimaAtmos $name: $*"
    flock "$GPU_LOCK" "$JULIA" --project="$CA_ENV" "$HARNESS_DIR/ca_driver.jl" \
        --out "$out" "$@" > "$out/log.txt" 2>&1
    local rc=$?
    tail -n 3 "$out/log.txt"
    return $rc
}

# Batched ClimaAtmos runs: one Julia session (and one GPU-lock hold) for
# several runs, so JIT compilation is paid once per model type, not per run.
#   ca_batch_begin; ca_add NAME args...; ...; ca_batch_run LABEL
# Arguments must not contain whitespace.  --straka-ic runs must come after
# every stock density-current run (the IC is redefined in-session).
ca_batch_begin() { CA_BATCH=(); }
ca_add() {
    local name=$1; shift
    clear_run "$RUNS_DIR/$name"
    CA_BATCH+=("--out $RUNS_DIR/$name $*")
}
ca_batch_run() {
    local label=$1
    local file="$RUNS_DIR/ca_batch_$label.txt"
    printf '%s\n' "${CA_BATCH[@]}" > "$file"
    echo "== ClimaAtmos batch $label (${#CA_BATCH[@]} runs): $file"
    flock "$GPU_LOCK" "$JULIA" --project="$CA_ENV" "$HARNESS_DIR/ca_driver.jl" \
        --batch "$file" > "$RUNS_DIR/ca_batch_$label.log" 2>&1
    local rc=$?
    grep -E "ca_driver done|run failed" -A3 "$RUNS_DIR/ca_batch_$label.log" | tail -n 40
    return $rc
}

# woof_run NAME [driver args...]: one woof run under the GPU lock.
woof_run() {
    local name=$1; shift
    local out="$RUNS_DIR/$name"
    clear_run "$out"
    echo "== woof $name: $*"
    (cd "$WOOF_ROOT" && flock "$GPU_LOCK" "$WOOF_PY" -m \
        tools.compare.climaatmos_idealized.woof_driver --out "$out" "$@" \
        > "$out/log.txt" 2>&1)
    local rc=$?
    tail -n 3 "$out/log.txt"
    return $rc
}

summarize() {
    (cd "$WOOF_ROOT" && "$WOOF_PY" -m tools.compare.climaatmos_idealized.analysis \
        summarize "$RUNS_DIR")
}
