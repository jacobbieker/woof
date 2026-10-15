# Source me: environment for the woof vs ClimaAtmos idealized benchmark.
#
# Everything ClimaAtmos-related lives under BENCH_ROOT (default
# /home/jacob/wf-scratch/idealized-benchmark): the Julia 1.11 tarball, the
# depot (packages + precompile cache), the ClimaAtmos.jl clone and the
# run environment.  Override BENCH_ROOT / WOOF_PY before sourcing to relocate.

BENCH_ROOT="${BENCH_ROOT:-/home/jacob/wf-scratch/idealized-benchmark}"
export BENCH_ROOT
export JULIA="${JULIA:-$BENCH_ROOT/julia-1.11.7/bin/julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$BENCH_ROOT/depot}"
export CA_REPO="${CA_REPO:-$BENCH_ROOT/ClimaAtmos.jl}"
export CA_ENV="${CA_ENV:-$BENCH_ROOT/env}"
export RUNS_DIR="${RUNS_DIR:-$BENCH_ROOT/runs}"
# /tmp is RAM-backed on the benchmark host: keep scratch on disk.
export TMPDIR="${TMPDIR_BENCH:-$BENCH_ROOT/tmp}"
mkdir -p "$TMPDIR" "$RUNS_DIR"

# GPU sharing: the card is shared with a running forecast.  Cap both models.
export CLIMACOMMS_DEVICE="${CLIMACOMMS_DEVICE:-CUDA}"
export JULIA_CUDA_HARD_MEMORY_LIMIT="${JULIA_CUDA_HARD_MEMORY_LIMIT:-6GiB}"
export CUPY_GPU_MEMORY_LIMIT="${CUPY_GPU_MEMORY_LIMIT:-6442450944}"
export GPU_LOCK="${GPU_LOCK:-/tmp/woof-gpu-test.lock}"

# Python for woof (any env with woof's GPU deps + numpy).
export WOOF_PY="${WOOF_PY:-python}"
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HARNESS_DIR
export WOOF_ROOT="${WOOF_ROOT:-$(cd "$HARNESS_DIR/../../.." && pwd)}"
export PYTHONPATH="$WOOF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
