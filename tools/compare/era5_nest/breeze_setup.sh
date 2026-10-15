#!/usr/bin/env bash
# Install Julia 1.12.6 and the Breeze/NumericalEarth environment the MC3E
# timing driver runs in, entirely under $BENCH_ROOT (default
# ~/wf-scratch/regional-benchmark).  Network and CPU only; no GPU is touched.
set -euo pipefail
BENCH_ROOT=${BENCH_ROOT:-$HOME/wf-scratch/regional-benchmark}
NE_COMMIT=${NE_COMMIT:-66112eac}
JULIA_VERSION=1.12.6
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$BENCH_ROOT/julia"
cd "$BENCH_ROOT/julia"
if [ ! -x "julia-$JULIA_VERSION/bin/julia" ]; then
    curl -sSL -o julia.tar.gz \
        "https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
    tar xzf julia.tar.gz
fi
if [ ! -d ne/src ]; then
    curl -sSL -o ne.tar.gz "https://codeload.github.com/NumericalEarth/NumericalEarth.jl/tar.gz/$NE_COMMIT"
    mkdir -p ne && tar xzf ne.tar.gz -C ne --strip-components=1
fi
mkdir -p env
cp "$HERE/breeze/Project.toml" env/Project.toml
export JULIA_DEPOT_PATH="$BENCH_ROOT/julia/depot"
export TMPDIR=${TMPDIR:-$BENCH_ROOT/tmp}
mkdir -p "$TMPDIR"
"./julia-$JULIA_VERSION/bin/julia" --project=env -e '
    using Pkg
    Pkg.develop(path="ne")
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.status()'
