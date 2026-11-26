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
# routines; run_coupling_prep.F90 the column package's coupling preparation
# (seaice_column_coupling_prep and seaice_column_scale_fluxes, cut verbatim);
# run_variational_basis.F90 the variational basis set-up (the wachspress, pwl,
# variational_shared and numerics modules verbatim; asin and tan the host libm's).
# run_velocity_weak.F90 the weak-form solver (mpas_seaice_velocity_solver_weak.F
# verbatim, with seaice_normal_vectors cut verbatim; atan2, asin, sin, cos and
# tan the host libm's, matmul libgfortran's).
# The objects must contain no FMA instruction (checked with objdump
# when it is present).  expf (RKIND 4) and exp (RKIND 8), which ice_strength
# calls, are the host glibc's, or tools/oracle_libm's transcriptions of them on
# a host whose libm is not glibc.  The column-package oracles (run_column,
# run_init_thermo; README.md) are built at RKIND 8 only, in BUILD_DIR/column:
# the column package is dbl_kind and mpas_seaice_column.F does not compile at
# RKIND 4; its atan, cos, exp, log, pow and tanh are the host libm's.  gfortran is the
# mpas-oracle pixi environment's unless GFORTRAN names one.
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
# WOOF_ORACLE_PATCH names a patch in this directory: an opt-in correction's
# oracle (zsal_solve.patch; the fixture's PROVENANCE.md), built from the checked
# tree copied into BUILD_DIR/source and patched there
if [[ -n "${WOOF_ORACLE_PATCH:-}" ]]; then
    rm -rf "${build_dir}/source" && mkdir -p "${build_dir}/source"
    cp -R "${source_root}/src" "${build_dir}/source/"
    patch -s -d "${build_dir}/source" -p1 < "${script_dir}/${WOOF_ORACLE_PATCH}.patch"
    source_root="${build_dir}/source"
fi

seaice="${source_root}/src/core_seaice"
# seaice_mesh's interpolation routines and the routines the variational basis
# set-up calls (seaice_cell_vertices_at_vertex, seaice_project_3D_vector_onto_local_2D
# with seaice_dot_product_3space and the private local_eastern_and_northern_unit_vectors,
# seaice_grid_rotation_forward) and the weak form's (seaice_normal_vectors and
# the private routines it calls), byte for byte, inside a module of that name
mesh_cut="${build_dir}/seaice_mesh_interpolation.F90"
{
    printf 'module seaice_mesh\n  use mpas_derived_types\n  use mpas_pool_routines\n  implicit none\n'
    printf '  private\n  public :: seaice_interpolate_cell_to_vertex, seaice_interpolate_vertex_to_cell, &\n'
    printf '            seaice_cell_vertices_at_vertex, seaice_dot_product_3space, seaice_grid_rotation_forward, &\n'
    printf '            seaice_project_3D_vector_onto_local_2D, seaice_normal_vectors\ncontains\n'
    awk '/^  subroutine seaice_interpolate_cell_to_vertex\(/{on=1} on{print} /^  end subroutine seaice_interpolate_vertex_to_cell/{on=0}' \
        "${seaice}/shared/mpas_seaice_mesh.F"
    for routine in seaice_cell_vertices_at_vertex seaice_dot_product_3space seaice_project_3D_vector_onto_local_2D \
                   local_eastern_and_northern_unit_vectors seaice_grid_rotation_forward \
                   seaice_normal_vectors seaice_normal_vectors_polygon seaice_normal_vectors_triangle \
                   normal_vectors_planar_polygon normal_vectors_planar_triangle \
                   normal_vectors_spherical_polygon_metric normal_vectors_spherical_triangle_metric \
                   cross_product_3space; do
        awk -v r="${routine}" '$0 ~ "^ *subroutine +" r "\\(" {on=1} on {print} on && $0 ~ "^ *end subroutine +" r "([^a-zA-Z_0-9]|$)" {on=0}' \
            "${seaice}/shared/mpas_seaice_mesh.F"
    done
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
    # the time-step oracle, on the framework stand-ins the ocean oracles share
    # (tools/mpas_ocean_oracle/stubs.F90: pools with time levels, the time
    # manager, the self-describing stream), in a directory of its own
    mkdir -p step
    # the host libm shim's objects (non-glibc hosts) sit one directory up
    step_libm=()
    for object in ${libm_objects[@]+"${libm_objects[@]}"}; do
        [[ ${object} == /* ]] && step_libm+=("${object}") || step_libm+=("../${object}")
    done
    ( cd step
      cut() {  # cut FILE ROUTINE...: the routines, verbatim
          local file=$1; shift
          for routine in "$@"; do
              awk -v r="${routine}" '$0 ~ "^ *subroutine +" r "\\(" {on=1} on {print} on && $0 ~ "^ *end subroutine +" r "([^a-zA-Z_0-9]|$)" {on=0}' "${file}"
          done
      }
      "${fc[@]}" "${flags[@]}" "${source_root}/src/framework/mpas_kind_types.F"
      "${fc[@]}" "${flags[@]}" "${script_dir}/../mpas_ocean_oracle/stubs.F90"
      "${fc[@]}" "${flags[@]}" "${script_dir}/../mpas_ocean_oracle/oracle_stream.F90"
      "${fc[@]}" "${flags[@]}" "${mesh_cut}"
      "${fc[@]}" "${flags[@]}" "${seaice}/column/ice_kinds_mod.F90"
      "${fc[@]}" "${flags[@]}" "${seaice}/column/constants/cice/ice_constants_colpkg.F90"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_constants.F"
      "${fc[@]}" "${flags[@]}" "${script_dir}/step_standins.F90"
      "${fc[@]}" "${flags[@]}" "${script_dir}/colpkg_standin.F90"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_constitutive_relation.F"
      # the variational basis set-up, verbatim
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_numerics.F"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_variational_shared.F"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_wachspress.F"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_pwl.F"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_variational.F"
      # the weak-form solver, verbatim
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_weak.F"
      { cat "${script_dir}/diagnostics_cut_head.F90"
        cut "${seaice}/shared/mpas_seaice_diagnostics.F" seaice_initialize_time_diagnostics \
            seaice_set_time_diagnostics seaice_check_state seaice_load_balance_timers
        echo "end module seaice_diagnostics"; } > seaice_diagnostics_cut.F
      "${fc[@]}" "${flags[@]}" seaice_diagnostics_cut.F
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver.F"
      # incremental remap, verbatim (the remap module with -Dprivate=public, a
      # visibility change only, so run_incremental_remap can call its stages);
      # rbf_standins.F90: the RBF setup (the coefficients are the input's);
      # remap_standins.F90:
      # seaice_error's codes and abort write, and upwind, which cannot run in v8.4.2
      "${fc[@]}" "${flags[@]}" "${script_dir}/rbf_standins.F90"
      "${fc[@]}" "${flags[@]}" "${script_dir}/remap_standins.F90"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_advection_incremental_remap_tracers.F"
      "${fc[@]}" "${flags[@]}" -Dprivate=public "${seaice}/shared/mpas_seaice_advection_incremental_remap.F"
      "${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_advection.F"
      { cat "${script_dir}/column_cut_head.F90"
        cut "${seaice}/shared/mpas_seaice_column.F" seaice_column_predynamics_time_integration \
            seaice_column_dynamics_time_integration seaice_column_postdynamics_time_integration \
            seaice_column_reinitialize_fluxes seaice_column_reinitialize_atmospheric_fluxes \
            seaice_column_reinitialize_oceanic_fluxes seaice_column_reinitialize_diagnostics_thermodynamics \
            seaice_column_reinitialize_diagnostics_bgc seaice_column_reinitialize_diagnostics_dynamics
        cat "${script_dir}/column_cut_tail.F90"; } > seaice_column_cut.F
      "${fc[@]}" "${flags[@]}" seaice_column_cut.F
      { cat "${script_dir}/timestep_cut_head.F90"
        cut "${seaice}/shared/mpas_seaice_time_integration.F" seaice_timestep store_timestep_counter \
            seaice_timestep_finalize
        echo "end module seaice_time_integration"; } > seaice_time_integration_cut.F
      "${fc[@]}" "${flags[@]}" seaice_time_integration_cut.F
      "${fc[@]}" "${flags[@]}" "${script_dir}/run_seaice_step.F90"
      "${fc[@]}" "${flags[@]}" "${script_dir}/run_incremental_remap.F90"
      # the coupling preparation, cut verbatim (coupling_cut_head.F90)
      { cat "${script_dir}/coupling_cut_head.F90"
        cut "${seaice}/shared/mpas_seaice_column.F" seaice_column_coupling_prep seaice_column_scale_fluxes
        echo "end module seaice_column_coupling"; } > seaice_column_coupling_cut.F
      "${fc[@]}" "${flags[@]}" seaice_column_coupling_cut.F
      "${fc[@]}" "${flags[@]}" "${script_dir}/run_coupling_prep.F90"
      "${fc[@]}" "${flags[@]}" "${script_dir}/run_variational_basis.F90"
      "${fc[@]}" "${flags[@]}" "${script_dir}/run_velocity_weak.F90"
      if command -v objdump >/dev/null; then
          if objdump -d mpas_seaice_*.o seaice_*.o | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
              echo "a time-step oracle object contains an FMA instruction; refusing" >&2
              exit 4
          fi
      fi
      "${fc[@]}" -o "${build_dir}/run_seaice_step_${precision}" run_seaice_step.o seaice_time_integration_cut.o \
          seaice_column_cut.o mpas_seaice_velocity_solver.o seaice_diagnostics_cut.o \
          mpas_seaice_advection.o mpas_seaice_advection_incremental_remap.o \
          mpas_seaice_advection_incremental_remap_tracers.o remap_standins.o rbf_standins.o \
          mpas_seaice_velocity_solver_variational.o mpas_seaice_velocity_solver_constitutive_relation.o \
          mpas_seaice_velocity_solver_wachspress.o mpas_seaice_velocity_solver_pwl.o \
          mpas_seaice_velocity_solver_variational_shared.o mpas_seaice_numerics.o \
          mpas_seaice_velocity_solver_weak.o step_standins.o colpkg_standin.o mpas_seaice_constants.o ice_constants_colpkg.o ice_kinds_mod.o \
          seaice_mesh_interpolation.o oracle_stream.o stubs.o mpas_kind_types.o \
          ${step_libm[@]+"${step_libm[@]}"}
      "${fc[@]}" -o "${build_dir}/run_coupling_prep_${precision}" run_coupling_prep.o seaice_column_coupling_cut.o \
          mpas_seaice_constants.o ice_constants_colpkg.o ice_kinds_mod.o oracle_stream.o stubs.o mpas_kind_types.o
      "${fc[@]}" -o "${build_dir}/run_variational_basis_${precision}" run_variational_basis.o \
          mpas_seaice_velocity_solver_variational.o mpas_seaice_velocity_solver_wachspress.o \
          mpas_seaice_velocity_solver_pwl.o mpas_seaice_velocity_solver_variational_shared.o mpas_seaice_numerics.o \
          mpas_seaice_velocity_solver_constitutive_relation.o \
          seaice_mesh_interpolation.o mpas_seaice_constants.o ice_constants_colpkg.o ice_kinds_mod.o \
          oracle_stream.o stubs.o mpas_kind_types.o
      "${fc[@]}" -o "${build_dir}/run_velocity_weak_${precision}" run_velocity_weak.o \
          mpas_seaice_velocity_solver_weak.o mpas_seaice_velocity_solver_constitutive_relation.o \
          seaice_mesh_interpolation.o mpas_seaice_constants.o ice_constants_colpkg.o ice_kinds_mod.o \
          oracle_stream.o stubs.o mpas_kind_types.o ${step_libm[@]+"${step_libm[@]}"}
      "${fc[@]}" -o "${build_dir}/run_incremental_remap_${precision}" run_incremental_remap.o \
          mpas_seaice_advection_incremental_remap.o mpas_seaice_advection_incremental_remap_tracers.o \
          remap_standins.o rbf_standins.o seaice_diagnostics_cut.o oracle_stream.o stubs.o mpas_kind_types.o
      # the init oracle: the same driver and remap objects with MPAS's own RBF set-up
      # (mpas_rbf_interp_initialize = mpas_initialize_vectors, mpas_init_reconstruct and
      # what they call), cut verbatim out of the pinned operator sources into modules of
      # those names, so the init derives coeffs_reconstruct itself
      rm -rf init && mkdir init && ( cd init
        cutr() {  # cutr FILE ROUTINE...: the subroutines and functions, verbatim (any case, any prefix)
            local file=$1; shift
            for routine in "$@"; do
                awk -v r="${routine}" '{ l = tolower($0) }
                    l ~ "^ *(real[^ ]* +)?(subroutine|function) +" tolower(r) " *\\(" {on=1}
                    on {print}
                    on && l ~ "^ *end +(subroutine|function) +" tolower(r) "([^a-z_0-9]|$)" {on=0}' "${file}"
            done
        }
        ops="${source_root}/src/operators"
        { printf 'module mpas_vector_operations\n  use mpas_kind_types\n  use mpas_derived_types\n'
          printf '  use mpas_pool_routines\n  implicit none\n  public\ncontains\n'
          cutr "${ops}/mpas_vector_operations.F" mpas_unit_vec_in_r3 mpas_cross_product_in_r3 \
              mpas_initialize_vectors mpas_fix_periodicity
          printf 'end module mpas_vector_operations\n'
          printf 'module mpas_rbf_interpolation\n  use mpas_kind_types\n  use mpas_derived_types\n'
          printf '  use mpas_pool_routines\n  use mpas_vector_operations\n  implicit none\n  private\n'
          printf '  public :: mpas_rbf_interp_initialize, mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs\n'
          printf 'contains\n'
          cutr "${ops}/mpas_rbf_interpolation.F" mpas_rbf_interp_initialize \
              mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs evaluate_rbf \
              mpas_set_up_vector_dirichlet_rbf_matrix_and_rhs mpas_legs elgs
          printf 'end module mpas_rbf_interpolation\n'
          printf 'module mpas_vector_reconstruction\n  use mpas_kind_types\n  use mpas_derived_types\n'
          printf '  use mpas_pool_routines\n  use mpas_rbf_interpolation\n  use mpas_vector_operations\n'
          printf '  implicit none\n  private\n  public :: mpas_init_reconstruct\ncontains\n'
          cutr "${ops}/mpas_vector_reconstruction.F" mpas_init_reconstruct
          printf 'end module mpas_vector_reconstruction\n'; } > rbf_cut.F
        "${fc[@]}" "${flags[@]}" -I.. rbf_cut.F
        if command -v objdump >/dev/null && objdump -d rbf_cut.o | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
            echo "the init oracle's RBF object contains an FMA instruction; refusing" >&2
            exit 4
        fi
        "${fc[@]}" -o "${build_dir}/run_incremental_remap_init_${precision}" ../run_incremental_remap.o \
            ../mpas_seaice_advection_incremental_remap.o ../mpas_seaice_advection_incremental_remap_tracers.o \
            ../remap_standins.o rbf_cut.o ../seaice_diagnostics_cut.o ../oracle_stream.o ../stubs.o \
            ../mpas_kind_types.o ) )
done
# The column oracle, at RKIND 8 only: the column package is dbl_kind and
# mpas_seaice_column.F hands it RKIND arrays, so the wrappers do not compile at
# RKIND 4 (REAL(4) actual arguments to REAL(8) dummies).  The column package
# (column/*.F90, the stand-alone build's cice constants, as column/Makefile
# copies them) and mpas_seaice_column.F are compiled verbatim, the latter with
# -Dprivate=public (a visibility change only: run_column reads the tracer
# object and calls the init wrappers); seaice_init's state routines are cut
# verbatim out of mpas_seaice_initialize.F (initialize_cut_head.F90 and
# initialize_cut_tail.F90 hold the module head and stand-ins that stop by name
# for the initial conditions this oracle does not run), the coupler-flux reset
# out of mpas_seaice_forcing.F.  Built on the shared framework stand-ins in a
# directory of its own.  tanh, pow and cos (colpkg_init_itd,
# colpkg_init_thermo) are the host glibc's.
rm -rf "${build_dir}/column" && mkdir -p "${build_dir}/column" && cd "${build_dir}/column"
flags=(-c -cpp -O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
column="${seaice}/column"
cut() {  # cut FILE ROUTINE...: the routines, verbatim
    local file=$1; shift
    for routine in "$@"; do
        awk -v r="${routine}" '$0 ~ "^ *subroutine +" r "\\(" {on=1} on {print} on && $0 ~ "^ *end subroutine +" r "([^a-zA-Z_0-9]|$)" {on=0}' "${file}"
    done
}
"${fc[@]}" "${flags[@]}" "${source_root}/src/framework/mpas_kind_types.F"
"${fc[@]}" "${flags[@]}" "${script_dir}/../mpas_ocean_oracle/stubs.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/../mpas_ocean_oracle/oracle_stream.F90"
"${fc[@]}" "${flags[@]}" "${column}/ice_kinds_mod.F90"
"${fc[@]}" "${flags[@]}" "${column}/ice_warnings.F90"
"${fc[@]}" "${flags[@]}" "${column}/constants/cice/ice_constants_colpkg.F90"
for module in ice_colpkg_shared ice_therm_shared ice_orbital ice_mushy_physics ice_colpkg_tracers ice_zbgc_shared \
              ice_therm_mushy ice_atmo ice_age ice_firstyear ice_flux_colpkg ice_meltpond_cesm ice_meltpond_lvl \
              ice_meltpond_topo ice_therm_bl99 ice_therm_0layer ice_therm_vertical ice_itd ice_aerosol ice_brine \
              ice_zsalinity ice_zbgc ice_therm_itd ice_algae ice_shortwave ice_mechred ice_colpkg; do
    "${fc[@]}" "${flags[@]}" "${column}/${module}.F90"
done
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_constants.F"
"${fc[@]}" "${flags[@]}" "${script_dir}/remap_standins.F90"
"${fc[@]}" "${flags[@]}" -Dprivate=public "${seaice}/shared/mpas_seaice_column.F"
{ cat "${script_dir}/initialize_cut_head.F90"
  cut "${seaice}/shared/mpas_seaice_initialize.F" init_ice_state init_ice_state_uniform_ice init_ice_cice_default \
      initial_category_areas_and_volumes init_ice_velocity_uniform initialize_coupler_fields
  cat "${script_dir}/initialize_cut_tail.F90"; } > seaice_initialize_cut.F
"${fc[@]}" "${flags[@]}" seaice_initialize_cut.F
{ printf 'module seaice_forcing\n  use mpas_derived_types\n  use mpas_pool_routines\n  implicit none\n'
  printf '  private\n  public :: seaice_reset_coupler_fluxes\ncontains\n'
  cut "${seaice}/shared/mpas_seaice_forcing.F" seaice_reset_coupler_fluxes reset_atmospheric_coupler_fluxes \
      reset_ocean_coupler_fluxes
  printf 'end module seaice_forcing\n'; } > seaice_forcing_cut.F
"${fc[@]}" "${flags[@]}" seaice_forcing_cut.F
# the time step with the column package (run_column's seaice_step stage): the
# velocity solver, incremental remap and seaice_timestep compiled or cut
# verbatim as for run_seaice_step, here on the column package itself (its
# colpkg_ice_strength) and the verbatim mpas_seaice_column.F
"${fc[@]}" "${flags[@]}" "${mesh_cut}"
"${fc[@]}" "${flags[@]}" "${script_dir}/step_standins.F90"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_constitutive_relation.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_numerics.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_variational_shared.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_wachspress.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_pwl.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_variational.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver_weak.F"
{ cat "${script_dir}/diagnostics_cut_head.F90"
  cut "${seaice}/shared/mpas_seaice_diagnostics.F" seaice_initialize_time_diagnostics \
      seaice_set_time_diagnostics seaice_check_state seaice_load_balance_timers
  echo "end module seaice_diagnostics"; } > seaice_diagnostics_cut.F
"${fc[@]}" "${flags[@]}" seaice_diagnostics_cut.F
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_velocity_solver.F"
"${fc[@]}" "${flags[@]}" "${script_dir}/rbf_standins.F90"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_advection_incremental_remap_tracers.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_advection_incremental_remap.F"
"${fc[@]}" "${flags[@]}" "${seaice}/shared/mpas_seaice_advection.F"
{ cat "${script_dir}/timestep_cut_head.F90"
  cut "${seaice}/shared/mpas_seaice_time_integration.F" seaice_timestep store_timestep_counter \
      seaice_timestep_finalize
  echo "end module seaice_time_integration"; } > seaice_time_integration_cut.F
"${fc[@]}" "${flags[@]}" seaice_time_integration_cut.F
"${fc[@]}" "${flags[@]}" "${script_dir}/run_column.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/run_init_thermo.F90"
column_libm=()
while read -r object; do column_libm+=("${object}"); done \
    < <(bash "${script_dir}/../oracle_libm/build_host_libm.sh" 2> libm.txt)
if command -v objdump >/dev/null; then
    if objdump -d ice_*.o mpas_seaice_*.o seaice_*.o | grep -Eq '\bv?fn?m(add|sub)([0-9]|[[:space:]]|$)'; then
        echo "a column oracle object contains an FMA instruction; refusing" >&2
        exit 4
    fi
fi
"${fc[@]}" -o "${build_dir}/run_column_r8" run_column.o seaice_forcing_cut.o seaice_initialize_cut.o \
    seaice_time_integration_cut.o mpas_seaice_velocity_solver.o seaice_diagnostics_cut.o \
    mpas_seaice_advection.o mpas_seaice_advection_incremental_remap.o \
    mpas_seaice_advection_incremental_remap_tracers.o rbf_standins.o \
    mpas_seaice_velocity_solver_variational.o mpas_seaice_velocity_solver_constitutive_relation.o \
    mpas_seaice_velocity_solver_wachspress.o mpas_seaice_velocity_solver_pwl.o \
    mpas_seaice_velocity_solver_variational_shared.o mpas_seaice_numerics.o \
    mpas_seaice_velocity_solver_weak.o step_standins.o seaice_mesh_interpolation.o \
    mpas_seaice_column.o remap_standins.o mpas_seaice_constants.o ice_*.o oracle_stream.o stubs.o \
    mpas_kind_types.o ${column_libm[@]+"${column_libm[@]}"}
"${fc[@]}" -o "${build_dir}/run_init_thermo_r8" run_init_thermo.o ice_*.o \
    ${column_libm[@]+"${column_libm[@]}"}
cd "${build_dir}"
{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none; r4 adds -DSINGLE_PRECISION;"
    echo "       mpas_seaice_velocity_solver.F adds -Dprivate=public; the column oracle (RKIND 8 only)"
    echo "       builds mpas_seaice_column.F with -Dprivate=public"
    echo "host: $(uname -srm); libm: $(ldd --version 2>/dev/null | head -1)"
    echo "expf/exp: $(sed 's/^libm: //' "${build_dir}/r8/libm.txt")"
    echo "fma: $(grep -m1 -ow fma /proc/cpuinfo 2>/dev/null || echo absent)"
} > "${build_dir}/TOOLCHAIN.txt"
echo "built ${build_dir}/run_oracle_r{4,8} ${build_dir}/run_subcycle_r{4,8} ${build_dir}/run_presubcycle_r{4,8}" \
     "${build_dir}/run_seaice_step_r{4,8} ${build_dir}/run_incremental_remap_r{4,8}" \
     "${build_dir}/run_incremental_remap_init_r{4,8}" "${build_dir}/run_coupling_prep_r{4,8}" \
     "${build_dir}/run_variational_basis_r{4,8}" "${build_dir}/run_velocity_weak_r{4,8}" \
     "${build_dir}/run_column_r8" \
     "${build_dir}/run_init_thermo_r8"
