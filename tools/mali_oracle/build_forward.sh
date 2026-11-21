#!/usr/bin/env bash
# Build the MALI v8.4.2 forward-mode oracle (run_forward.F90) at the strict
# recipe, at RKIND 4 and RKIND 8.
#
#   bash tools/mali_oracle/build_forward.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# As build.sh's oracles, plus the forward mode's own code, all compiled
# verbatim and checked against SOURCES.sha256: mpas_li_core.F,
# mpas_li_time_integration.F, the framework's mpas_timekeeping.F with
# mpas_string_utils.F, and the ESMF time library it wraps
# (src/external/esmf_time_f90, -DHIDE_MPI).  stubs.F90 is compiled with
# -DMALI_REAL_TIMEKEEPING (its stand-in mpas_timekeeping gives way to the real
# one), and stubs_forward.F90 adds the stream manager, RBF, decomposition,
# analysis-member and statistics stand-ins mpas_li_core.F USEs.  The
# reconstruction cut gains a no-op mpas_init_reconstruct: the coefficients
# are the input mesh's.  The objects must contain no FMA instruction.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_forward.sh MPAS_SOURCE_ROOT BUILD_DIR" >&2
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
esmf="${src}/external/esmf_time_f90"
esmf_units=(ESMF_BaseMod ESMF_BaseTimeMod ESMF_FractionMod ESMF_CalendarMod ESMF_ShrTimeMod ESMF_Stubs
            ESMF_TimeIntervalMod ESMF_TimeMod ESMF_AlarmMod ESMF_ClockMod ESMF_AlarmClockMod MeatMod ESMF
            wrf_error_fatal wrf_message)
units=(framework/mpas_constants operators/mpas_vector_operations operators/mpas_matrix_operations
       operators/mpas_geometry_utils core_landice/shared/mpas_li_constants core_landice/shared/mpas_li_setup
       core_landice/shared/mpas_li_mask core_landice/mode_forward/mpas_li_thermal
       core_landice/mode_forward/mpas_li_diagnostic_vars core_landice/mode_forward/mpas_li_sia
       core_landice/mode_forward/mpas_li_advection core_landice/mode_forward/mpas_li_calving
       core_landice/mode_forward/mpas_li_velocity_simple core_landice/mode_forward/mpas_li_velocity
       core_landice/mode_forward/mpas_li_subglacial_hydro core_landice/mode_forward/mpas_li_time_integration_fe
       core_landice/mode_forward/mpas_li_time_integration core_landice/mode_forward/mpas_li_core)
reconstruct_cut="${build_dir}/mpas_vector_reconstruction_cut.F"
{
    printf 'module mpas_vector_reconstruction\n  use mpas_derived_types\n  use mpas_pool_routines\n  use mpas_threading\n'
    printf '#define MPAS_ACC_TIMER_START(X)\n#define MPAS_ACC_TIMER_STOP(X)\n  implicit none\n  private\n'
    printf '  public :: mpas_reconstruct, mpas_init_reconstruct\n  interface mpas_reconstruct\n     module procedure mpas_reconstruct_1d\n'
    printf '     module procedure mpas_reconstruct_2d\n  end interface\ncontains\n'
    printf '  subroutine mpas_init_reconstruct(meshPool, includeHalos)\n'
    printf '    ! the mesh initialisation: coeffs_reconstruct is the input mesh'"'"'s\n'
    printf '    type (mpas_pool_type), intent(in) :: meshPool\n    logical, optional, intent(in) :: includeHalos\n'
    printf '  end subroutine mpas_init_reconstruct\n'
    awk '/^  subroutine mpas_reconstruct_2d\(/{on=1} on{print} /^  end subroutine mpas_reconstruct_2d!/{on=0}' \
        "${src}/operators/mpas_vector_reconstruction.F"
    awk '/^  subroutine mpas_reconstruct_1d\(/{on=1} on{print} /^  end subroutine mpas_reconstruct_1d!/{on=0}' \
        "${src}/operators/mpas_vector_reconstruction.F"
    printf 'end module mpas_vector_reconstruction\n'
} > "${reconstruct_cut}"
for precision in r4 r8; do
    rm -rf "${build_dir}/${precision}" && mkdir -p "${build_dir}/${precision}" && cd "${build_dir}/${precision}"
    flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
    [[ ${precision} == r4 ]] && flags+=(-DSINGLE_PRECISION)
    objects=()
    for unit in "${esmf_units[@]}"; do
        "${fc[@]}" -c -cpp -O0 -ffp-contract=off -DHIDE_MPI -I"${esmf}" "${esmf}/${unit}.F90"
        objects+=("${unit}.o")
    done
    "${fc[@]}" "${flags[@]}" "${src}/framework/mpas_kind_types.F"
    "${fc[@]}" "${flags[@]}" -DMALI_REAL_TIMEKEEPING -I"${src}/framework" "${script_dir}/stubs.F90"
    "${fc[@]}" "${flags[@]}" "${src}/framework/mpas_string_utils.F"
    "${fc[@]}" "${flags[@]}" "${src}/framework/mpas_timekeeping.F"
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs_forward.F90"
    "${fc[@]}" "${flags[@]}" "${reconstruct_cut}"
    objects+=(stubs.o mpas_string_utils.o mpas_timekeeping.o stubs_forward.o mpas_vector_reconstruction_cut.o)
    for unit in "${units[@]}"; do
        "${fc[@]}" "${flags[@]}" "${src}/${unit}.F"
        objects+=("$(basename "${unit}").o")
    done
    "${fc[@]}" "${flags[@]}" "${script_dir}/run_forward.F90"
    libm_objects=()
    while read -r object; do libm_objects+=("${object}"); done \
        < <(bash "${script_dir}/../oracle_libm/build_host_libm.sh" 2> libm.txt)
    if command -v objdump >/dev/null; then
        if objdump -d "${objects[@]}" | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
            echo "an oracle object contains an FMA instruction; refusing" >&2
            exit 4
        fi
    fi
    "${fc[@]}" -o "${build_dir}/run_forward_${precision}" run_forward.o "${objects[@]}" \
        ${libm_objects[@]+"${libm_objects[@]}"} mpas_kind_types.o
done
libm_note=$(sed 's/^libm: //' "${build_dir}/r4/libm.txt")
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none;" \
         "-DSINGLE_PRECISION for RKIND 4, none for RKIND 8; ESMF time library -DHIDE_MPI"
    echo "host: $(uname -srm); libm: $(ldd --version 2>/dev/null | head -1)"
    echo "libm: ${libm_note}"
    echo "fma: $(grep -m1 -ow fma /proc/cpuinfo 2>/dev/null || echo absent)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built run_forward_r4 and run_forward_r8 in ${build_dir}"
