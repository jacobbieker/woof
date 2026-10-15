#!/usr/bin/env bash
# Run the small coupled atmosphere–ocean experiment (CPU only).
#
#   OC=/path/to/scratch tools/compare/ocean_coupling/run.sh [case ...]
#
# OC holds the Julia install and depot (see README.md). Results land in $OC_OUT
# (default: $OC/results). Cases default to all six.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OC:?set OC to the scratch directory holding julia-1.12.7/ and depot/}"
export JULIA_DEPOT_PATH="$OC/depot"
export TMPDIR="$OC/tmp"
export OC_OUT="${OC_OUT:-$OC/results}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$OC_OUT"
exec "$OC/julia-1.12.7/bin/julia" --project="$here" "$here/coupled_regional_box.jl" "$@"
