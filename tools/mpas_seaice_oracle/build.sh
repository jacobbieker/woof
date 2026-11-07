#!/usr/bin/env bash
# Build the MPAS-Seaice v8.4.2 variational EVP internal-stress oracle at the
# strict recipe.
#
#   bash tools/mpas_seaice_oracle/build.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# MPAS_SOURCE_ROOT is an MPAS-Model v8.4.2 tree (f34984b2).  The two seaice
# modules and the kind module are compiled verbatim (checked against
# SOURCES.sha256) with gfortran at -O0 -ffp-contract=off and MPAS's default
# single precision (-DSINGLE_PRECISION, RKIND = 4).  stubs.F90 supplies the
# framework types the routines walk; run_oracle.F90 drives them.  The objects
# must contain no FMA instruction (checked with objdump when it is present).
# gfortran is the mpas-oracle pixi environment's unless GFORTRAN names one.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build.sh MPAS_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(cd "$1" && pwd)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -r -a fc <<< "${GFORTRAN:-gfortran}"

( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | { command -v sha256sum >/dev/null && sha256sum -c --quiet - || shasum -a 256 -c --quiet -; } ) || {
    echo "an MPAS source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

rm -rf "${build_dir}/r4" && mkdir -p "${build_dir}/r4" && cd "${build_dir}/r4"
flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none -DSINGLE_PRECISION)
"${fc[@]}" "${flags[@]}" "${source_root}/src/framework/mpas_kind_types.F"
"${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
"${fc[@]}" "${flags[@]}" "${source_root}/src/core_seaice/shared/mpas_seaice_velocity_solver_constitutive_relation.F"
"${fc[@]}" "${flags[@]}" "${source_root}/src/core_seaice/shared/mpas_seaice_velocity_solver_variational.F"
"${fc[@]}" "${flags[@]}" "${script_dir}/run_oracle.F90"
objects=(run_oracle.o mpas_seaice_velocity_solver_variational.o \
         mpas_seaice_velocity_solver_constitutive_relation.o stubs.o mpas_kind_types.o)
if command -v objdump >/dev/null; then
    if objdump -d mpas_seaice_velocity_solver_variational.o mpas_seaice_velocity_solver_constitutive_relation.o \
        | grep -Eq '\bv?fn?m(add|sub)[0-9]'; then
        echo "an oracle object contains an FMA instruction; refusing" >&2
        exit 4
    fi
fi
"${fc[@]}" -o "${build_dir}/run_oracle_r4" "${objects[@]}"
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none -DSINGLE_PRECISION (RKIND 4)"
    echo "host: $(uname -srm)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_oracle_r4"
