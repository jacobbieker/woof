# MALI v8.4.2 forward-mode oracle

Written by `tools/mali_oracle/build_forward.sh` (`run_forward_r4`, `_r8`) and
`make_forward_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-21.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified
  (`tools/mali_oracle/SOURCES.sha256`). Beyond the forward-Euler step's
  oracle it compiles `mpas_li_core.F` and `mpas_li_time_integration.F`
  whole, the framework's `mpas_timekeeping.F` with `mpas_string_utils.F`,
  and the ESMF time library it wraps (`src/external/esmf_time_f90`,
  `-DHIDE_MPI`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, with `-DSINGLE_PRECISION` for RKIND 4 and without for
  RKIND 8; no object contains an FMA instruction.
- Driver: `run_forward.F90` reads a self-describing stream (every config,
  every field by pool, the text fields and `vertexMask`'s second time level),
  builds a one-block domain, and runs a cold start as `mpas_core_init` and
  `mpas_core_run` do: `li_simulation_clock_init`, `li_core_init`, and
  `li_core_run` to the clock's stop time. It writes the error words of the
  three (all zero here), the count of logged errors, the clock's final time
  and every field.
- The stand-ins (`stubs.F90` with `-DMALI_REAL_TIMEKEEPING`,
  `stubs_forward.F90`): the stream manager reads and writes nothing (the
  driver puts the input stream's fields in the pools), RBF and
  reconstruction initialisation keep the input mesh's coefficients, every
  analysis member is off, the statistics stop if reached. Halo exchanges
  and one-rank reductions are the identity. Scratch fields resolve to arrays
  the driver fills with random words, and every diagnostic the run sets
  starts as random words, so no output may depend on them.
- The fixture is x86-64 gfortran's. Its `MAX`/`MIN` keep an operand on a tie
  of signed zeros (`maxss`/`minss`); arm64 gfortran (`fmaxnm`) orders `-0`
  below `+0`, so an oracle rebuilt on Apple silicon differs in the
  hydrology's zero water thicknesses' signs only.
- `stubs.F90`'s pool `put` routines now take the entry's slot before the
  pointer assignment (gfortran evaluated the slot function, which adds the
  entry, twice). Every other MALI oracle was rebuilt with it on pb the same
  day and wrote its committed fixture byte for byte.

## Coverage

The MALI test cases' physics on the periodic hexagonal meshes of
`woof/verify/mali_hexmesh.py` (20 x 20 cells, 4 km; the variable resolution
mesh stretches the cell spacing from about 1.75 to 7.2 km). Every case runs
on both meshes, at both precisions.

| case | levels | velocity | thermal | clock | steps |
|---|---|---|---|---|---|
| `dome` | 5 | sia (barycentric tangent) | none | 30 days for 120, noleap | 4 |
| `dome_adaptive` | 5 | sia (kite areas) | none | adaptive (CFL fraction 0.004, 8-day maximum, 20-day force interval), 40 days | 6 (uniform), 10 (varres) |
| `dome_calving` | 5 | sia (normal slope) | none | 10 days for 40; marine bed, 'floating' calving | 4 |
| `eismint2_a_temperature` | 10 | sia, PB1982 | temperature (air temperature and SMB from file) | 5 days for 20 from 2000-02-20, gregorian (over the leap day) | 4 |
| `eismint2_a_enthalpy` | 10 | sia, CP2010 | enthalpy | 30 days for 120 | 4 |
| `hydro_radial` | 5 | none (radial prescribed velocity) | none | 6 h for 1 day; `config_SGH` | 4 |
| `hydro_marine` | 5 | none | none | 6 h for 12 h; grounded ice below sea level | 2 |
| `enthalpy_benchmark` | 10 | none | enthalpy, 1000 m slab | 1 year for 4 | 4 |

The only logged errors are `mpas_calculate_barycentric_weights`' weight sums
across the periodic seam (229 and 195 at RKIND 4, 76 at RKIND 8, in the
barycentric cases), which v8.4.2 logs and does not raise.

Two v8.4.2 quirks run in these cases with the model's words:

- `li_SGH_init` reads a cell mask that holds only `li_calculate_mask_init`'s
  initial-extent bit, so grounded ice below sea level (`hydro_marine`) gets
  the ocean's water pressure. The port's opt-in
  `woof_sgh_init_mask = 'corrected'` computes the mask first.
- With the enthalpy solver, `tracer_setup`/`tracer_finish` in
  `mpas_li_advection.F` convert with Kelvin temperatures where the thermal
  solver's conversions take Celsius, so the advected ice turns temperate and
  melts away within the run (`eismint2_a_enthalpy`).

Compared at both precisions: every mesh, geometry, thermal, velocity and
hydro field (the mesh's own inputs included, which must come back
unchanged), `xtime`, `simulationStartTime`, `timestepNumber`, `deltat`,
`daysSinceStart`, the CFL limits and the clock's final time. At RKIND 8 the
fixture leaves out `eTheta` (glibc's binary64 `atan`).
