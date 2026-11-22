# MPAS-Seaice v8.4.2 time step with the variational basis computed in the init

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_step_remap_fixture.py --basis` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21. Toolchain, source pin and the verbatim modules as in
`../mpas_seaice_step_remap/PROVENANCE.md`, with the variational basis set-up
compiled verbatim (`../mpas_seaice_variational_basis/PROVENANCE.md`).

`run_seaice_step.F90` with the config `oracle_compute_basis`: after
`seaice_init_evp` it calls `seaice_init_velocity_solver_variational`, as
`seaice_init_velocity_solver` does, so the basis comes from the mesh (its
inputs are noise); then N steps with the velocity solver and incremental remap.
The npz holds the fields whose words the run changed.

| case | mesh | basis | steps | notes |
|---|---|---|---|---|
| `basis_step_hex_wachspress` | planar hexagons | Wachspress, Dunavant 8 | 3 | metric terms off |
| `basis_step_quad_pwl_two_subcycles` | planar quadrilaterals | PWL | 2 | revised EVP, two dynamics subcycles |
| `basis_step_sphere_hex_north_wachspress_metric` | sphere, North Pole | Wachspress | 3 | metric terms on (RKIND 8: glibc 2.44's asin/tan) |
| `basis_step_sphere_quad_midlat_varres_pwl_unrotated` | sphere, mid-latitude, stretched | PWL | 2 | grid not rotated, metric terms on |
| `basis_step_sphere_hex_south_fekete6_no_metric` | sphere, South Pole | Wachspress, Fekete 6 | 2 | metric terms off |
