#!/usr/bin/env bash
# Build the MALI v8.4.2 oracles (SIA velocity solver; geometry diagnostics;
# thermal initialisation) at the strict recipe.
#
#   bash tools/mali_oracle/build.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# MPAS_SOURCE_ROOT is an MPAS-Model v8.4.2 tree (f34984b2).  The land-ice
# modules and the framework operators they call are compiled verbatim
# (checked against SOURCES.sha256) with gfortran at -O0 -ffp-contract=off and
# MPAS's default single precision (-DSINGLE_PRECISION, RKIND = 4).  stubs.F90
# supplies the framework types, logging and abort the modules USE;
# run_oracle.F90 makes one li_sia_solve call; run_geometry.F90 makes the
# mask, geometry, layer-thickness and layer-velocity calls; run_thermal.F90
# makes the thermal init and enthalpy-conversion calls.  The objects must contain no FMA
# instruction (checked with objdump when it is present).  powf and expf are
# the host libm's; TOOLCHAIN.txt records the glibc and whether the CPU has
# FMA.  On a host whose libm is not glibc (macOS), the oracles link
# glibc_flt32_host.c instead: the repository's MIT glibc-float32 powf/expf
# transcription, compiled for the host by CC (default cc), so they write the
# glibc host's words.  gfortran is the mpas-oracle pixi environment's unless
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

rm -rf "${build_dir}/r4" && mkdir -p "${build_dir}/r4" && cd "${build_dir}/r4"
flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none -DSINGLE_PRECISION)
src="${source_root}/src"
"${fc[@]}" "${flags[@]}" "${src}/framework/mpas_kind_types.F"
"${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
units=(framework/mpas_constants operators/mpas_vector_operations operators/mpas_matrix_operations
       operators/mpas_geometry_utils core_landice/shared/mpas_li_constants core_landice/shared/mpas_li_setup
       core_landice/shared/mpas_li_mask core_landice/mode_forward/mpas_li_thermal
       core_landice/mode_forward/mpas_li_diagnostic_vars core_landice/mode_forward/mpas_li_sia
       core_landice/mode_forward/mpas_li_advection)
objects=()
for unit in "${units[@]}"; do
    "${fc[@]}" "${flags[@]}" "${src}/${unit}.F"
    objects+=("$(basename "${unit}").o")
done
"${fc[@]}" "${flags[@]}" "${script_dir}/run_oracle.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/run_geometry.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/run_thermal.F90"
libm_objects=()
libm_note="host glibc ($(getconf GNU_LIBC_VERSION 2>/dev/null || true))"
if ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    read -r -a cc <<< "${CC:-cc}"
    "${cc[@]}" -c -O0 -ffp-contract=off -I"${script_dir}/../../woof/core/kernels" \
        "${script_dir}/glibc_flt32_host.c" -o glibc_flt32_host.o
    libm_objects=(glibc_flt32_host.o)
    libm_note="glibc_flt32_host.c (woof/core/kernels/glibc_flt32.cuh; host libm is not glibc)"
fi
if command -v objdump >/dev/null; then
    if objdump -d "${objects[@]}" | grep -Eq '\bv?fn?m(add|sub)([0-9]|\b)'; then
        echo "an oracle object contains an FMA instruction; refusing" >&2
        exit 4
    fi
fi
"${fc[@]}" -o "${build_dir}/run_oracle_r4" run_oracle.o "${objects[@]}" ${libm_objects[@]+"${libm_objects[@]}"} stubs.o mpas_kind_types.o
"${fc[@]}" -o "${build_dir}/run_geometry_r4" run_geometry.o "${objects[@]}" ${libm_objects[@]+"${libm_objects[@]}"} stubs.o mpas_kind_types.o
"${fc[@]}" -o "${build_dir}/run_thermal_r4" run_thermal.o "${objects[@]}" ${libm_objects[@]+"${libm_objects[@]}"} stubs.o mpas_kind_types.o
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none -DSINGLE_PRECISION (RKIND 4)"
    echo "host: $(uname -srm); libm: $(ldd --version 2>/dev/null | head -1)"
    echo "powf/expf: ${libm_note}"
    echo "fma: $(grep -m1 -ow fma /proc/cpuinfo 2>/dev/null || echo absent)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_oracle_r4 ${build_dir}/run_geometry_r4 ${build_dir}/run_thermal_r4"
