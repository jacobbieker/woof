# MPAS-Seaice v8.4.2 time step with the weak-form stress divergence

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_step_remap_fixture.py --weak` on pb (Linux 6.12 x86_64, AVX2 and FMA,
glibc 2.44), 2026-11-21. Toolchain, source pin and the verbatim modules as in
`../mpas_seaice_step_remap/PROVENANCE.md`, with
`mpas_seaice_velocity_solver_weak.F` compiled verbatim and the normal-vector
routines cut verbatim (`../mpas_seaice_weak/PROVENANCE.md`; libgfortran's
`matmul` and glibc 2.44's libm as described there).

`run_seaice_step.F90` with `config_stress_divergence_scheme = 'weak'` and the
config `oracle_compute_basis`. After `seaice_init_evp` it calls
`seaice_init_velocity_solver_weak`, as `seaice_init_velocity_solver` does, so
the normal vectors and rotated latitudes come from the mesh (their inputs are
noise). Then come N steps with the velocity solver (weak form) and
incremental remap. The npz holds the fields whose words the run changed.

| case | mesh | steps | notes |
|---|---|---|---|
| `weak_step_hex` | planar hexagons | 3 | |
| `weak_step_quad_revised_two_subcycles` | planar quadrilaterals | 2 | revised EVP, two dynamics subcycles |
| `weak_step_sphere_hex_north` | sphere, North Pole | 3 | glibc 2.44's libm |
| `weak_step_sphere_quad_midlat_varres_unrotated` | sphere, mid-latitude, stretched | 2 | grid not rotated |
