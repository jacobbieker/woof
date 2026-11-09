#!/usr/bin/env bash
# Build the MPAS-Seaice v8.4.2 oracles (variational EVP internal stress; the
# EVP velocity solve; the velocity solver's pre-subcycle routines) at the
# strict recipe, at RKIND 4 and RKIND 8.
#
#   bash tools/mpas_seaice_oracle/build.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# MPAS_SOURCE_ROOT is an MPAS-Model v8.4.2 tree (f34984b2).  The seaice
# modules, the column constants and the kind module are compiled verbatim
# (checked against SOURCES.sha256) with gfortran at -O0 -ffp-contract=off,
# once at MPAS's default single precision (-DSINGLE_PRECISION, RKIND = 4) and
# once at RKIND = 8 (the column package is double, so production MPAS-Seaice
# is built double).  mpas_seaice_velocity_solver.F is compiled with
# -Dprivate=public, which changes only which routines the driver may call.
# seaice_mesh's two interpolation routines (seaice_interpolate_cell_to_vertex,
# seaice_interpolate_vertex_to_cell) are cut verbatim out of the pinned
# mpas_seaice_mesh.F into a module of that name; the rest of that file pulls
# in the stream manager and is not built.  stubs.F90 supplies the framework
# types the routines walk; run_oracle.F90 drives the internal stress,
# run_subcycle.F90 the velocity solve and run_presubcycle.F90 the pre-subcycle
# routines.  The objects must contain no FMA instruction (checked with objdump
# when it is present).  expf (RKIND 4) and exp (RKIND 8), which ice_strength
# calls, are the host glibc's, or tools/oracle_libm's transcriptions of them on
# a host whose libm is not glibc.  gfortran is the mpas-oracle pixi
# environment's unless GFORTRAN names one.
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

seaice="${source_root}/src/core_seaice"
# seaice_mesh's interpolation routines, byte for byte, inside a module of that name
mesh_cut="${build_dir}/seaice_mesh_interpolation.F90"
{
    printf 'module seaice_mesh\n  use mpas_derived_types\n  use mpas_pool_routines\n  implicit none\n'
    printf '  private\n  public :: seaice_interpolate_cell_to_vertex, seaice_interpolate_vertex_to_cell\ncontains\n'
    awk '/^  subroutine seaice_interpolate_cell_to_vertex\(/{on=1} on{print} /^  end subroutine seaice_interpolate_vertex_to_cell/{on=0}' \
        "${seaice}/shared/mpas_seaice_mesh.F"
    printf 'end module seaice_mesh\n'
} > "${mesh_cut}"
for precision in r4 r8; do
    rm -rf "${build_dir}/${precision}" && mkdir -p "${build_dir}/${precision}" && cd "${build_dir}/${precision}"
    flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
    [[ ${precision} == r4 ]] && flags+=(-DSINGLE_PRECISION)
    "${fc[@]}" "${flags[@]}" "${source_root}/src/framework/mpas_kind_types.F"
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
    "${fc[@]}" "${flags[@]}" "${mesh_cut}"
    "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_constitutive_relation.F"
    "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_variational.F"
    # the stand-alone (non-ESM) build's column constants, as column/Makefile copies them
    "${fc[@]}" "${flags[@]}" "${seaice}/column/ice_kinds_mod.F90"
    "${fc[@]}" "${flags[@]}" "${seaice}/column/constants/cice/ice_constants_colpkg.F90"
    "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_constants.F"
    # visibility only: the routines the driver calls are private to the module
    "${fc[@]}" "${flags[@]}" -Dprivate=public "${seaice}/shared/mpas_seaice_velocity_solver.F"
    "${fc[@]}" "${flags[@]}" "${script_dir}/run_oracle.F90"
    "${fc[@]}" "${flags[@]}" "${script_dir}/run_subcycle.F90"
    "${fc[@]}" "${flags[@]}" "${script_dir}/run_presubcycle.F90"
    seaice_objects=(mpas_seaice_velocity_solver.o mpas_seaice_constants.o ice_constants_colpkg.o ice_kinds_mod.o \
                    mpas_seaice_velocity_solver_variational.o mpas_seaice_velocity_solver_constitutive_relation.o \
                    seaice_mesh_interpolation.o)
    libm_objects=()
    while read -r object; do libm_objects+=("${object}"); done \
        < <(bash "${script_dir}/../oracle_libm/build_host_libm.sh" 2> libm.txt)
    if command -v objdump >/dev/null; then
        if objdump -d "${seaice_objects[@]}" | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
            echo "an oracle object contains an FMA instruction; refusing" >&2
            exit 4
        fi
    fi
    "${fc[@]}" -o "${build_dir}/run_oracle_${precision}" run_oracle.o "${seaice_objects[@]}" stubs.o mpas_kind_types.o
    "${fc[@]}" -o "${build_dir}/run_subcycle_${precision}" run_subcycle.o "${seaice_objects[@]}" stubs.o mpas_kind_types.o
    "${fc[@]}" -o "${build_dir}/run_presubcycle_${precision}" run_presubcycle.o "${seaice_objects[@]}" \
        ${libm_objects[@]+"${libm_objects[@]}"} stubs.o mpas_kind_types.o
done
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none; r4 adds -DSINGLE_PRECISION;"
    echo "       mpas_seaice_velocity_solver.F adds -Dprivate=public"
    echo "host: $(uname -srm); libm: $(ldd --version 2>/dev/null | head -1)"
    echo "expf/exp: $(sed 's/^libm: //' "${build_dir}/r8/libm.txt")"
    echo "fma: $(grep -m1 -ow fma /proc/cpuinfo 2>/dev/null || echo absent)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_oracle_r{4,8} ${build_dir}/run_subcycle_r{4,8} ${build_dir}/run_presubcycle_r{4,8}"
