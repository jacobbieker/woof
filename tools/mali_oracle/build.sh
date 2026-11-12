#!/usr/bin/env bash
# Build the MALI v8.4.2 oracles (SIA velocity solver; geometry diagnostics;
# thermal initialisation; advection; thermal solver) at the strict recipe, at
# RKIND 4 and RKIND 8.
#
#   bash tools/mali_oracle/build.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# MPAS_SOURCE_ROOT is an MPAS-Model v8.4.2 tree (f34984b2).  The land-ice
# modules and the framework operators they call are compiled verbatim
# (checked against SOURCES.sha256) with gfortran at -O0 -ffp-contract=off,
# once at MPAS's default single precision (-DSINGLE_PRECISION, RKIND = 4,
# BUILD_DIR/*_r4) and once at RKIND = 8 (BUILD_DIR/*_r8).  stubs.F90
# supplies the framework types, logging and abort the modules USE;
# run_oracle.F90 makes one li_sia_solve call; run_geometry.F90 makes the
# mask, geometry, layer-thickness and layer-velocity calls; run_thermal.F90
# makes the thermal init and enthalpy-conversion calls; run_advection.F90 makes
# li_advection_thickness_tracers' call after a restart's thermal init;
# run_thermal_solver.F90 makes li_basal_melt_floating_ice's and
# li_thermal_solver's calls after a restart's thermal init;
# run_time_integration_fe.F90 makes one li_time_integrator_forwardeuler
# after a restart's thermal init.  The objects must contain no FMA
# instruction (checked with objdump when it is present).  powf, expf, cosf,
# sinf, atanf and tanhf (RKIND 4) and pow, exp and tanh (RKIND 8) are the host libm's; TOOLCHAIN.txt records the glibc and whether the CPU has
# FMA.  On a host whose libm is not glibc (macOS), the oracles link
# tools/oracle_libm/glibc_host.cpp instead: the repository's MIT glibc
# transcriptions, compiled for the host by CXX (default c++), so they write
# the glibc host's words.  gfortran is the mpas-oracle pixi environment's unless
# GFORTRAN names one.
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

src="${source_root}/src"
units=(framework/mpas_constants operators/mpas_vector_operations operators/mpas_matrix_operations
       operators/mpas_geometry_utils core_landice/shared/mpas_li_constants core_landice/shared/mpas_li_setup
       core_landice/shared/mpas_li_mask core_landice/mode_forward/mpas_li_thermal
       core_landice/mode_forward/mpas_li_diagnostic_vars core_landice/mode_forward/mpas_li_sia
       core_landice/mode_forward/mpas_li_advection core_landice/mode_forward/mpas_li_calving
       core_landice/mode_forward/mpas_li_velocity_simple core_landice/mode_forward/mpas_li_velocity
       core_landice/mode_forward/mpas_li_subglacial_hydro core_landice/mode_forward/mpas_li_time_integration_fe)
drivers=(run_oracle run_geometry run_thermal run_advection run_thermal_solver run_time_integration_fe)
# mpas_reconstruct_1d and _2d, cut verbatim out of the pinned
# mpas_vector_reconstruction.F (the rest of it, mpas_init_reconstruct's RBF
# fit, is the mesh's initialisation: the drivers supply coeffs_reconstruct)
reconstruct_cut="${build_dir}/mpas_vector_reconstruction_cut.F"
{
    printf 'module mpas_vector_reconstruction\n  use mpas_derived_types\n  use mpas_pool_routines\n  use mpas_threading\n'
    printf '#define MPAS_ACC_TIMER_START(X)\n#define MPAS_ACC_TIMER_STOP(X)\n  implicit none\n  private\n'
    printf '  public :: mpas_reconstruct\n  interface mpas_reconstruct\n     module procedure mpas_reconstruct_1d\n'
    printf '     module procedure mpas_reconstruct_2d\n  end interface\ncontains\n'
    awk '/^  subroutine mpas_reconstruct_2d\(/{on=1} on{print} /^  end subroutine mpas_reconstruct_2d!/{on=0}' \
        "${source_root}/src/operators/mpas_vector_reconstruction.F"
    awk '/^  subroutine mpas_reconstruct_1d\(/{on=1} on{print} /^  end subroutine mpas_reconstruct_1d!/{on=0}' \
        "${source_root}/src/operators/mpas_vector_reconstruction.F"
    printf 'end module mpas_vector_reconstruction\n'
} > "${reconstruct_cut}"
for precision in r4 r8; do
    rm -rf "${build_dir}/${precision}" && mkdir -p "${build_dir}/${precision}" && cd "${build_dir}/${precision}"
    flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
    [[ ${precision} == r4 ]] && flags+=(-DSINGLE_PRECISION)
    "${fc[@]}" "${flags[@]}" "${src}/framework/mpas_kind_types.F"
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
    "${fc[@]}" "${flags[@]}" "${reconstruct_cut}"
    objects=(mpas_vector_reconstruction_cut.o)
    for unit in "${units[@]}"; do
        "${fc[@]}" "${flags[@]}" "${src}/${unit}.F"
        objects+=("$(basename "${unit}").o")
    done
    for driver in "${drivers[@]}"; do
        "${fc[@]}" "${flags[@]}" "${script_dir}/${driver}.F90"
    done
    libm_objects=()
    while read -r object; do libm_objects+=("${object}"); done \
        < <(bash "${script_dir}/../oracle_libm/build_host_libm.sh" 2> libm.txt)
    if command -v objdump >/dev/null; then
        if objdump -d "${objects[@]}" | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
            echo "an oracle object contains an FMA instruction; refusing" >&2
            exit 4
        fi
    fi
    for driver in "${drivers[@]}"; do
        "${fc[@]}" -o "${build_dir}/${driver}_${precision}" "${driver}.o" "${objects[@]}" \
            ${libm_objects[@]+"${libm_objects[@]}"} stubs.o mpas_kind_types.o
    done
done
libm_note=$(sed 's/^libm: //' "${build_dir}/r4/libm.txt")
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none;" \
         "-DSINGLE_PRECISION for RKIND 4, none for RKIND 8"
    echo "host: $(uname -srm); libm: $(ldd --version 2>/dev/null | head -1)"
    echo "powf/expf/cosf/sinf/atanf/tanhf (RKIND 4), pow/exp/tanh (RKIND 8): ${libm_note}"
    echo "fma: $(grep -m1 -ow fma /proc/cpuinfo 2>/dev/null || echo absent)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${drivers[*]/%/_r4} and ${drivers[*]/%/_r8} in ${build_dir}"
