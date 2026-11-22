# MPAS-Seaice v8.4.2 weak-form stress divergence oracle

Written by `tools/mpas_seaice_oracle/build.sh` and `make_weak_fixture.py` on
pb (Linux 6.12 x86_64, AVX2 and FMA, glibc 2.44), 2026-11-21.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256` (it pins
  `mpas_seaice_velocity_solver_weak.F`).
- Compiled verbatim: `mpas_seaice_velocity_solver_weak.F`,
  `..._constitutive_relation.F`, the seaice and colpkg constants. Cut verbatim
  out of `mpas_seaice_mesh.F` into module `seaice_mesh`:
  `seaice_normal_vectors`, `seaice_normal_vectors_polygon`,
  `seaice_normal_vectors_triangle`, `normal_vectors_planar_polygon`,
  `normal_vectors_planar_triangle`, `normal_vectors_spherical_polygon_metric`,
  `normal_vectors_spherical_triangle_metric`, `cross_product_3space`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`. No FMA instruction in the objects. The intrinsic `matmul`,
  which the spherical normal vectors call, is libgfortran's, and on this host
  it is FMA-contracted (`dest = fma(a, b, dest)` from 0) at both precisions.
  The port's `_matvec` does the same. libm (sphere only): the host glibc
  2.44's `asin`, `atan2`, `sin`, `cos` and `tan` at each precision. `asinf`,
  `atan2f` and `tanf` round correctly; `sinf`, `cosf` and the binary64
  functions do not. The port's `metric_libm="host"` gives these words on a
  glibc 2.44 host.
- Driver: `run_velocity_weak.F90`. It runs `seaice_init_evp`, then
  `seaice_init_velocity_solver_weak`, then `oracle_n_iterations` passes of
  `seaice_internal_stress_weak`, then `seaice_final_divergence_shear_weak`,
  and writes every pool back.
- `inputs.sha256` pins the stream bytes of every case at both precisions.
  Every velocity_weak array is noise on input. The npz holds the fields whose
  words the run changed.

## Cases (7 per precision)

| case | mesh | iterations | notes |
|---|---|---|---|
| `weak_planar_hex` | planar hexagons | 3 | |
| `weak_planar_quad_revised_column` | planar quadrilaterals | 4 | revised EVP, column package (ridging fields) |
| `weak_planar_hex_unrotated_column` | planar hexagons | 2 | grid not rotated, column package |
| `weak_sphere_hex_north` | sphere, North Pole | 3 | column package |
| `weak_sphere_quad_south_revised` | sphere, South Pole | 3 | revised EVP |
| `weak_sphere_hex_midlat_varres_unrotated` | sphere, mid-latitude, stretched | 2 | grid not rotated, column package |
| `weak_sphere_quad_midlat_rotated_column` | sphere, mid-latitude | 2 | column package |

`ridgeShear` carries v8.4.2's quirk. `Delta` is assigned as a whole array in
each pass of the loop over cells, so every cell's ridgeShear uses the last
owned cell's Delta. The port keeps that by default; the opt-in correction
`weak_ridge_delta` uses each cell's own Delta.
