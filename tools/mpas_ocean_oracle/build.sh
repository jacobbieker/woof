#!/usr/bin/env bash
# Build the MPAS-Ocean v8.4.2 oracles (run_oracle: four tendency terms;
# run_pgrad_hmix: pressure gradient and horizontal momentum mixing;
# run_tendency: ocn_tend_vel/thick; run_eos: the equation of state;
# run_diagnostics: ocn_diagnostic_solve; run_tracer_advection: the tracer
# advection; run_tracer_hmix: horizontal tracer mixing; run_tend_tracer:
# ocn_tend_tracer; run_vmix: implicit vertical mixing; run_vert_transport:
# ocn_vert_transport_velocity_top) at the
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
    "${fc[@]}" "${flags[@]}" "${script_dir}/stubs_unported.F90"
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
    # replace stubs_tendency.F90's ocn_equation_of_state (and ocn_diagnostics)
    # there.  ocn_diagnostic_solve, ocn_diagnostics_init and
    # ocn_compute_land_ice_flux_input_fields are cut verbatim out of
    # mpas_ocn_diagnostics.F (awk ranges) into diagnostics_cut_head.F90's module.
    mkdir -p eos
    ( cd eos
      for unit in mpas_ocn_equation_of_state_linear mpas_ocn_equation_of_state_jm mpas_ocn_equation_of_state \
                  mpas_ocn_diagnostics_routines mpas_ocn_thick_ale; do
          "${fc[@]}" "${flags[@]}" -I.. "${ocean}/${unit}.F"
      done
      { cat "${script_dir}/diagnostics_cut_head.F90"
        for routine in ocn_diagnostic_solve ocn_diagnostics_init ocn_compute_land_ice_flux_input_fields \
                       ocn_compute_KPP_input_fields ocn_vert_transport_velocity_top; do
            awk -v r="${routine}" '$0 ~ "^ *subroutine " r "\\(" {on=1} on {print} on && $0 ~ "^ *end subroutine " r "[^a-zA-Z_]" {on=0}' \
                "${ocean}/mpas_ocn_diagnostics.F"
        done
        echo "end module ocn_diagnostics"; } > ocn_diagnostics_cut.F
      "${fc[@]}" "${flags[@]}" -I.. ocn_diagnostics_cut.F
      # tracer advection: the two vertical stencils cut out of the operators'
      # helpers, then the three advection modules verbatim
      { cat "${script_dir}/tracer_advection_helpers_head.F90"
        for fn in mpas_tracer_advection_vflux4 mpas_tracer_advection_vflux3; do
            awk -v r="${fn}" '$0 ~ "function " r "\\(" {on=1} on {print} on && $0 ~ "^ *end function" {on=0}' \
                "${source_root}/src/operators/mpas_tracer_advection_helpers.F"
        done
        echo "end module mpas_tracer_advection_helpers"; } > mpas_tracer_advection_helpers_cut.F
      "${fc[@]}" "${flags[@]}" -I.. mpas_tracer_advection_helpers_cut.F
      for unit in mpas_ocn_tracer_advection_std mpas_ocn_tracer_advection_mono mpas_ocn_tracer_advection; do
          "${fc[@]}" "${flags[@]}" -I.. "${ocean}/${unit}.F"
      done
      for unit in mpas_ocn_tracer_hmix_del2 mpas_ocn_tracer_hmix_del4 mpas_ocn_tracer_hmix_redi mpas_ocn_tracer_hmix; do
          "${fc[@]}" "${flags[@]}" -I.. "${ocean}/${unit}.F"
      done
      # ocn_tend_tracer: the remaining tracer modules and the bulk forcing
      # verbatim, against the real equation of state, then mpas_ocn_tendency.F
      # itself against the real tracer modules (stubs_unported.F90 for the rest)
      for unit in mpas_ocn_tracer_nonlocalflux mpas_ocn_tracer_surface_flux_to_tend \
                  mpas_ocn_tracer_short_wave_absorption_jerlov mpas_ocn_tracer_short_wave_absorption \
                  mpas_ocn_surface_bulk_forcing mpas_ocn_tendency; do
          "${fc[@]}" "${flags[@]}" -I.. "${ocean}/${unit}.F"
      done
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/oracle_stream.F90"
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_eos.F90"
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_diagnostics.F90"
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_tracer_advection.F90"
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_tracer_hmix.F90"
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_tend_tracer.F90"
      # vertical mixing, after mpas_ocn_tendency.F (which links stubs_unported.F90's ocn_vmix)
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/stubs_vmix.F90"
      for unit in mpas_ocn_vmix_coefs_const mpas_ocn_vmix_coefs_tanh mpas_ocn_vmix_coefs_rich \
                  mpas_ocn_vmix_coefs_redi mpas_ocn_vmix; do
          "${fc[@]}" "${flags[@]}" -I.. "${ocean}/${unit}.F"
      done
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_vmix.F90"
      "${fc[@]}" "${flags[@]}" -I.. "${script_dir}/run_vert_transport.F90" )
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
            stubs_tendency.o stubs_unported.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    done
    "${fc[@]}" -o "${build_dir}/run_eos_${precision}" eos/run_eos.o eos/mpas_ocn_equation_of_state*.o \
        mpas_ocn_constants.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    "${fc[@]}" -o "${build_dir}/run_diagnostics_${precision}" eos/run_diagnostics.o eos/oracle_stream.o \
        eos/ocn_diagnostics_cut.o eos/mpas_ocn_diagnostics_routines.o eos/mpas_ocn_thick_ale.o eos/mpas_ocn_equation_of_state*.o \
        mpas_ocn_constants.o mpas_constants.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    "${fc[@]}" -o "${build_dir}/run_tracer_advection_${precision}" eos/run_tracer_advection.o eos/oracle_stream.o \
        eos/mpas_ocn_tracer_advection*.o eos/mpas_tracer_advection_helpers_cut.o mpas_ocn_constants.o \
        mpas_constants.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    "${fc[@]}" -o "${build_dir}/run_tracer_hmix_${precision}" eos/run_tracer_hmix.o eos/oracle_stream.o \
        eos/mpas_ocn_tracer_hmix*.o mpas_ocn_constants.o mpas_constants.o stubs.o mpas_kind_types.o \
        ${libm_objects[@]+"${libm_objects[@]}"}
    # the tendency module and every real module it reaches, from both directories
    "${fc[@]}" -o "${build_dir}/run_tend_tracer_${precision}" eos/run_tend_tracer.o eos/oracle_stream.o \
        eos/mpas_ocn_tendency.o eos/mpas_ocn_surface_bulk_forcing.o eos/mpas_ocn_tracer_*.o \
        eos/mpas_tracer_advection_helpers_cut.o eos/ocn_diagnostics_cut.o eos/mpas_ocn_diagnostics_routines.o eos/mpas_ocn_thick_ale.o \
        eos/mpas_ocn_equation_of_state*.o mpas_ocn_vel_*.o mpas_ocn_thick_*.o mpas_ocn_forcing.o mpas_ocn_constants.o \
        mpas_constants.o stubs_unported.o stubs.o mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    "${fc[@]}" -o "${build_dir}/run_vmix_${precision}" eos/run_vmix.o eos/oracle_stream.o eos/mpas_ocn_vmix*.o \
        eos/stubs_vmix.o eos/mpas_ocn_equation_of_state*.o mpas_ocn_constants.o mpas_constants.o stubs.o \
        mpas_kind_types.o ${libm_objects[@]+"${libm_objects[@]}"}
    "${fc[@]}" -o "${build_dir}/run_vert_transport_${precision}" eos/run_vert_transport.o eos/oracle_stream.o \
        eos/ocn_diagnostics_cut.o eos/mpas_ocn_diagnostics_routines.o eos/mpas_ocn_thick_ale.o \
        eos/mpas_ocn_equation_of_state*.o mpas_ocn_constants.o mpas_constants.o stubs.o mpas_kind_types.o \
        ${libm_objects[@]+"${libm_objects[@]}"}
done
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none; r4 adds -DSINGLE_PRECISION"
    echo "host: $(uname -srm)"
    cat "${build_dir}/r4/libm.txt"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_{oracle,pgrad_hmix,tendency,eos,diagnostics,tracer_advection,tracer_hmix,tend_tracer,vmix,vert_transport}_{r4,r8}"
