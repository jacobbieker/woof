# MALI v8.4.2 forward-mode oracle

Written by `tools/mali_oracle/build_forward.sh` (`run_forward_r4`, `_r8`) and
`make_forward_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-21.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified
  (`tools/mali_oracle/SOURCES.sha256`). Beyond the forward-Euler step's
  oracle it compiles `mpas_li_core.F`, `mpas_li_time_integration.F`,
  `mpas_li_statistics.F`, and the analysis-member driver with its
  `globalStats` and `regionalStats` members (`mpas_li_analysis_driver.F`,
  `mpas_li_global_stats.F`, `mpas_li_regional_stats.F`) whole, the framework's `mpas_timekeeping.F` with `mpas_string_utils.F`,
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
  and every field, then what `li_compute_statistics` leaves: the words its
  reductions and broadcasts return, in call order, and the log's messages;
  then each analysis-member compute's member, xtime and words (its pool's
  fields, taken when its `compute_` timer stops). Before `li_core_init` it
  makes MPAS's `li_analysis_setup_packages` call, with each member's
  configs at their Registry defaults when the input has none.
- The stand-ins (`stubs.F90` with `-DMALI_REAL_TIMEKEEPING`,
  `stubs_forward.F90`): the stream manager reads and writes nothing (the
  driver puts the input stream's fields in the pools), RBF and
  reconstruction initialisation keep the input mesh's coefficients; the
  streams an analysis member names are declared by the driver (`stream:NAME`
  configs, with their reference times) and no output stream's alarm rings,
  so a member at 'output_interval' computes on startup only. The pool stub
  iterates its entries in the order put, and `mpas_pool_get_package` (now a
  module procedure: its pointer argument needs the explicit interface)
  finds the packages the driver put. Halo exchanges and one-rank reductions are the
  identity; the reductions `li_compute_statistics` makes (sums, min/maxloc,
  broadcasts) also record each word they return, and `mpas_log_write` keeps
  every message but warnings and errors. Scratch fields resolve to arrays
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
| `simple_uniform` | 5 | simple, 'uniform' (1000 m/yr along x) | temperature | 30 days for 120 | 4 |
| `simple_radial_shelf` | 5 | simple, 'radial' | none | 10 days for 40; marine bed, 'floating' calving (the -880 m spike) | 4 |
| `simple_rim` | 5 | simple, 'uniform' | none | 10 days for 40; marine bed, 'topographic_threshold' calving (the deepened rim) | 4 |
| `stats_dome` | 5 | sia | none | 30 days for 120; statistics on startup and every 2nd step, column at the summit | 4 |
| `stats_eismint2` | 10 | sia, PB1982 | temperature | 10 days for 30; statistics every step, `config_stats_cell_ID` 0 (no column) | 3 |
| `stats_simple_shelf` | 5 | simple, 'radial' | none | 10 days for 30; marine bed, 'floating' calving; statistics on startup and every step | 3 |
| `stats_benchmark` | 10 | none (zero velocity) | enthalpy | 1 year for 2; statistics on startup and every 2nd step | 2 |
| `am_dome` | 5 | sia | none | 30 days for 120; globalStats every 70 days from a stream reference time of day 15 (and on startup), regionalStats every step ('dt', not on startup) over 3 regions (an inner disc, a half plane, none) | 4 |
| `am_marine` | 5 | sia (normal slope) | none | 10 days for 40; marine bed, 'floating' calving; globalStats at 'output_interval' (startup only), regionalStats every 20 days and on startup, the third region weighted 2 | 4 |

The 'simple' cases (added 2026-11-21 with the same build; every earlier
case's words came back byte for byte) also carry `xEdge`/`yEdge` (the edge
midpoints stored inside the rhombus, so seam edges take the model's
`cellsOnEdge(2)` branch), `normalVelocityInitial` and the higher-order
package's `dirichletVelocityMask`, which 'simple' never reads and must come
back unchanged.

The statistics cases (added 2026-11-21 with the build that compiles
`mpas_li_statistics.F`; every earlier case's words came back byte for byte)
carry `iceEdgeMask` in the scratch pool too. Their fixture entries add
`stats/real` and `stats/int` (the reductions' words) and `stats/log` (every
report from its rule line to its basal temperature line, a blank line
between). A zero velocity everywhere reports `+0`: `MAX(localMax,
-localMin)` compiles to `maxss` with `-localMin` as the destination, which
keeps `localMax` on a tie of signed zeros.

The analysis-member cases (added 2026-11-21 with the build that compiles the
driver and both members; every earlier case's words and input digests came
back byte for byte) carry the `globalStatsAM` and `regionalStatsAM` pools
(random words before the first compute), the `regions` pool's
`regionCellMasks` and the `nRegions` and `nRegionGroups` dimensions. Their
entries add `am/member` (1 globalStats, 2 regionalStats), `am/xtime` and
`am/words` for every compute in order. A mean over no ice is 0/0: x86-64's
default NaN (sign bit set) for the mean thickness and accumulation, and,
since gfortran compiles `-1.0 * a / b / c` as `-(a / b / c)`, the same NaN
with its sign bit clear for the melt rates.

The driver writes back the arrays it put, and the stub's
`mpas_pool_shift_time_levels` trades them between `vertexMask` and
`vertexMask@2`; `woof.verify.mali_forward_oracle.read_outputs` labels the
words by time level (after the initial solve's shift and one per step).
Every case but the fast-moving 'simple' ones holds the same words on both
levels at the end, so this changed no earlier word.

The only logged errors are `mpas_calculate_barycentric_weights`' weight sums
across the periodic seam (229 and 195 at RKIND 4, 76 at RKIND 8, in the
barycentric cases), which v8.4.2 logs and does not raise.

These v8.4.2 quirks run in these cases with the model's words:

- `li_SGH_init` reads a cell mask that holds only `li_calculate_mask_init`'s
  initial-extent bit, so grounded ice below sea level (`hydro_marine`) gets
  the ocean's water pressure. The port's opt-in
  `woof_sgh_init_mask = 'corrected'` computes the mask first.
- With the enthalpy solver, `tracer_setup`/`tracer_finish` in
  `mpas_li_advection.F` convert with Kelvin temperatures where the thermal
  solver's conversions take Celsius, so the advected ice turns temperate and
  melts away within the run (`eismint2_a_enthalpy`).
- `li_velocity_simple_block_init` writes its circular-shelf test's bed over
  the input's whenever calving is 'floating' or 'topographic_threshold'
  (`simple_radial_shelf`, `simple_rim`). The port's opt-in
  `woof_simple_velocity_bed = 'corrected'` keeps the input bed.
- The analysis members' means over no ice (or no grounded or floating ice)
  are 0/0, NaN, and the minimum thickness of no ice is the sentinel
  (100000 m globally, 10000 m in a region) (`am_dome`). The port's opt-in
  `woof_analysis_empty = 'corrected'` reports 0 for both.

Compared at both precisions: every mesh, geometry, thermal, velocity and
hydro field (the mesh's own inputs included, which must come back
unchanged), `xtime`, `simulationStartTime`, `timestepNumber`, `deltat`,
`daysSinceStart`, the CFL limits, the clock's final time, and in their
cases the statistics' and the analysis members' words. At RKIND 8 the
fixture leaves out `eTheta` (glibc's binary64 `atan`).
