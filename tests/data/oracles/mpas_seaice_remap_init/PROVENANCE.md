# MPAS-Seaice v8.4.2 incremental-remap oracle: the init geometry with MPAS's own RBF set-up

Written by `tools/mpas_seaice_oracle/build.sh` and `make_remap_fixture.py --init`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source, compiler and flags: as `tests/data/oracles/mpas_seaice_remap`
  (v8.4.2, `f34984b2`, gfortran 16.2.0 `-O0 -ffp-contract=off`,
  `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`).
- Driver: `run_incremental_remap.F90` linked as `run_incremental_remap_init_{r4,r8}`:
  instead of `rbf_standins.F90`, MPAS's own `mpas_initialize_vectors`,
  `mpas_rbf_interp_initialize`, `mpas_init_reconstruct` and the routines they
  call (`mpas_unit_vec_in_r3`, `mpas_cross_product_in_r3`,
  `mpas_fix_periodicity`, `mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs`,
  `evaluate_rbf`, `mpas_set_up_vector_dirichlet_rbf_matrix_and_rhs`,
  `mpas_legs`, `elgs`), cut verbatim out of `src/operators` (hashes in
  `tools/mpas_seaice_oracle/SOURCES.sha256`). `mpas_legs`/`elgs` are Tao
  Pang's elimination ("Copyright (c) Tao Pang 2001") as MPAS distributes it.
- Cases: `woof.verify.mpas_seaice_remap_init_oracle`: stage 5 (the init
  alone) and stage 4 (remap steps on that geometry), on planar hexagon and
  quadrilateral blocks (`woof.verify.mpas_seaice_planar_mesh`) and spherical
  ones (`woof.verify.mpas_seaice_sphere_mesh`) about either pole (rotated grid)
  and at mid-latitude (unrotated), uniform and variable-resolution. The mesh
  pool carries `is_periodic = .false.`, `x_period = y_period = 0`.
  `mpas_init_reconstruct` fits the owned cells only (`nCellsSolve`), so the
  halo's `coeffs_reconstruct` stays zero, as on one rank.
- `oracle-r{4,8}.npz`: per case, every output field whose words differ from the
  input's (the remap pool's geometry, the rotated coordinates,
  `localVerticalUnitVectors`, `edgeNormalVectors`, `cellTangentPlane` and
  `coeffs_reconstruct` included). `remap-logs.json`: the ordered tracer list,
  the warnings and the error code.
