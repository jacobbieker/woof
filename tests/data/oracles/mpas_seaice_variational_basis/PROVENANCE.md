# MPAS-Seaice v8.4.2 variational-basis oracle

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_variational_basis_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256`.
- Compiled verbatim: `mpas_seaice_velocity_solver_variational.F`,
  `..._variational_shared.F`, `..._wachspress.F`, `..._pwl.F`,
  `mpas_seaice_numerics.F`, `..._constitutive_relation.F`, the seaice and colpkg
  constants. Cut verbatim out of `mpas_seaice_mesh.F` into module `seaice_mesh`:
  `seaice_cell_vertices_at_vertex`, `seaice_dot_product_3space`,
  `seaice_project_3D_vector_onto_local_2D`,
  `local_eastern_and_northern_unit_vectors`, `seaice_grid_rotation_forward`
  (and the two interpolation routines the other oracles use).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`. No FMA instruction in any object. libm: the host glibc
  2.44's `asinf`/`tanf` (RKIND 4, correctly rounded) and `asin`/`tan` (RKIND 8,
  not correctly rounded: the port's `metric_libm="host"` gives these words on a
  glibc 2.44 host).
- Driver: `run_variational_basis.F90`. It calls
  `seaice_init_velocity_solver_variational` once with the case's configs and
  writes the velocity_variational pool back; MPAS_LOG_WARN lines go to the log
  (`<case>/log` in the npz).
- `inputs.sha256` pins the stream bytes of every case at both precisions. Every
  output array is noise on input.

## Cases (66 per precision)

- `<mesh>_wachspress` and `<mesh>_pwl` (Dunavant 8, rotated grid, metric terms
  on) on eight meshes: planar hexagons and quadrilaterals (jittered, culled
  land), planar polygons of 3 to 9 sides, and on the sphere hexagons about the
  North Pole, quadrilaterals about the South Pole, hexagons at mid-latitude
  (stretched), quadrilaterals about the North Pole (stretched) and polygons at
  random places (one cell where the rotated z is 0). The planar ones show
  v8.4.2's NaN metric terms (sphere_radius 0).
- `planar_polygons_<rule>` and `sphere_quad_south_<rule>`: every rule v8.4.2
  has (trapezoidal 1, 2, 5; Dunavant 1-10, 12, the odd orders with their WARN;
  Fekete 1-6, 8, 9), metric terms off.
- `sphere_polygons_<basis>_unrotated`, `sphere_hex_north_<basis>_no_metric`.
- `planar_hex_metric_radius`, `planar_polygons_metric_radius_unrotated`: the
  metric terms on a plane with a sphere_radius (asin of |x/R| below and above 1).
