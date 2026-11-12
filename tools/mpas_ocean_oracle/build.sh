#!/usr/bin/env bash
# Build the MPAS-Ocean v8.4.2 oracles (run_oracle: four tendency terms;
# run_pgrad_hmix: pressure gradient and horizontal momentum mixing;
# run_tendency: ocn_tend_vel/thick; run_eos: the equation of state) at the
# strict recipe, in both precisions.
#
#   bash tools/mpas_ocean_oracle/build.sh MPAS_SOURCE_ROOT BUILD_DIR
#
# MPAS_SOURCE_ROOT is an MPAS-Model v8.4.2 tree (f34984b2).  The ocean modules
# and the kind module are compiled verbatim (checked against SOURCES.sha256)
# with gfortran at -O0 -ffp-contract=off, twice: MPAS's default single
# precision (-DSINGLE_PRECISION, RKIND = 4 -> run_oracle_r4) and its double
# build (RKIND = 8 -> run_oracle_r8), the precision MPAS-Ocean is run at in
# practice.  stubs.F90 supplies the framework types the routines walk;
# run_oracle.F90 drives them.  The objects must contain no FMA instruction
# (checked with objdump when it is present).  gfortran is the mpas-oracle
# pixi environment's unless GFORTRAN names one.  On a host whose libm is not
# glibc the oracles link tools/oracle_libm/glibc_host.cpp, so their libm calls
# return glibc's words (TOOLCHAIN.txt records which).
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

ocean="${source_root}/src/core_ocean/shared"
for precision in r4 r8; do
    rm -rf "${build_dir}/${precision}" && mkdir -p "${build_dir}/${precision}" && cd "${build_dir}/${precision}"
    flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
    [[ ${precision} == r4 ]] && flags+=(-DSINGLE_PRECISION)
    "${fc[@]}" "${flags[@]}" "${source_root}/src/framework/mpas_kind_types.F"
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
    "${fc[@]}" "${flags[@]}" "${source_root}/src/framework/mpas_constants.F"
    "${fc[@]}" "${flags[@]}" "${ocean}/mpas_ocn_constants.F"
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs_tendency.F90"
    for unit in mpas_ocn_vel_coriolis mpas_ocn_vel_vadv mpas_ocn_thick_hadv mpas_ocn_thick_vadv \
                mpas_ocn_vel_pressure_grad mpas_ocn_vel_hmix_del2 mpas_ocn_vel_hmix_leith mpas_ocn_vel_hmix_del4 \
                mpas_ocn_vel_hmix mpas_ocn_forcing mpas_ocn_surface_bulk_forcing mpas_ocn_thick_surface_flux \
                mpas_ocn_vel_forcing_surface_stress mpas_ocn_vel_forcing_explicit_bottom_drag \
                mpas_ocn_vel_forcing_rayleigh mpas_ocn_vel_forcing mpas_ocn_tendency; do
        "${fc[@]}" "${flags[@]}" "${ocean}/${unit}.F"
    done
    for driver in run_oracle run_pgrad_hmix run_tendency; do
        "${fc[@]}" "${flags[@]}" "${script_dir}/${driver}.F90"
    done
    # the equation of state, verbatim, in a directory of its own: its modules
    # replace stubs_tendency.F90's ocn_equation_of_state there
    mkdir -p eos
    ( cd eos
      for unit in mpas_ocn_equation_of_state_linear mpas_ocn_equation_of_state_jm mpas_ocn_equation_of_state; do
          "${fc[@]}" "${flags[@]}" -I.. "${ocean}/${unit}.F"
      done
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_eos.F90" )
    if command -v objdump >/dev/null; then
        if objdump -d mpas_ocn_*.o eos/mpas_ocn_*.o mpas_constants.o | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
            echo "an oracle object contains an FMA instruction; refusing" >&2
            exit 4
        fi
    fi
    # glibc's expf/cosf/sinf/exp words: the host libm on glibc, else the
    # repository's transcriptions (tools/oracle_libm)
    libm_objects=()
    while read -r object; do libm_objects+=("${object}"); done \
        < <(bash "${script_dir}/../oracle_libm/build_host_libm.sh" 2> libm.txt)
    for driver in run_oracle run_pgrad_hmix run_tendency; do
        "${fc[@]}" -o "${build_dir}/${driver}_${precision}" "${driver}.o" mpas_ocn_*.o mpas_constants.o \
            stubs_tendency.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    done
    "${fc[@]}" -o "${build_dir}/run_eos_${precision}" eos/run_eos.o eos/mpas_ocn_equation_of_state*.o \
        mpas_ocn_constants.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
done
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none; r4 adds -DSINGLE_PRECISION"
    echo "host: $(uname -srm)"
    cat "${build_dir}/r4/libm.txt"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_{oracle,pgrad_hmix,tendency,eos}_{r4,r8}"
