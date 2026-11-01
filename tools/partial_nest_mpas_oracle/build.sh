#!/usr/bin/env bash
# Build the hex partial-band coupling-operator oracle against MPAS-Model v8.4.1.
#
#   bash tools/partial_nest_mpas_oracle/build.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# MPAS_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles.  extract.py cuts
# MPAS's own routines byte for byte; stubs.F90 supplies the kind, pool, log
# and nRelaxZone they USE; run_oracle.F90 drives them.  Two executables:
#   run_oracle_r8  RKIND = 8 (MPAS's default double build): the barycentric
#                  weights (woof.hex.partial_nest_coupling transcribes them in
#                  binary64);
#   run_oracle_r4  RKIND = 4 (the single build the hex dycore follows): the
#                  barycentric application and the relaxation-zone tendency.
# Reference build: -O0 -ffp-contract=off.  GFORTRAN defaults to
# `pixi exec -s gfortran gfortran`.  make_fixture.py writes the inputs, runs
# both and packs tests/data/oracles/partial_nest_mpas/fixture.npz.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build.sh MPAS_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(cd "$1" && pwd)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -r -a fc <<< "${GFORTRAN:-pixi exec -s gfortran gfortran}"
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | { command -v sha256sum >/dev/null && sha256sum -c --quiet - || shasum -a 256 -c --quiet -; } ) || {
    echo "an MPAS source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}/src"
for rk in r8 r4; do
    d="${build_dir}/${rk}"
    rm -rf "${d}" && mkdir -p "${d}" && cd "${d}"
    defs=()
    [[ ${rk} == r4 ]] && defs=(-DPN_SINGLE)
    flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none "${defs[@]}")
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
    "${fc[@]}" "${flags[@]}" "${build_dir}/src/mpas_vector_operations.F90"
    "${fc[@]}" "${flags[@]}" "${build_dir}/src/pn_mpas_geometry.F90"
    objs=(stubs.o mpas_vector_operations.o pn_mpas_geometry.o)
    if [[ ${rk} == r4 ]]; then
        "${fc[@]}" "${flags[@]}" "${build_dir}/src/pn_mpas_relax.F90"
        objs+=(pn_mpas_relax.o)
    fi
    "${fc[@]}" "${flags[@]}" "${script_dir}/run_oracle.F90"
    "${fc[@]}" -o "${build_dir}/run_oracle_${rk}" run_oracle.o "${objs[@]}"
done
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp (r8: RKIND 8; r4: -DPN_SINGLE, RKIND 4)"
    echo "host: $(uname -srm)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_oracle_r8 ${build_dir}/run_oracle_r4"
