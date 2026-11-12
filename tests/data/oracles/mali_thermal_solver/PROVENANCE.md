# MALI v8.4.2 thermal solver and basal melt oracle

Written by `tools/mali_oracle/build.sh` (`run_thermal_solver_r4`) and
`make_thermal_solver_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-09.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. It is the same
  12 files as the other MALI oracles (`tools/mali_oracle/SOURCES.sha256`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0. Reference of record:
  `-O0 -ffp-contract=off -DSINGLE_PRECISION` (RKIND = 4), and no object may
  contain an FMA instruction. The libm calls are glibc 2.44's `powf` (the
  heat dissipation's real power), `sinf` ('seroussi') and `tanhf`
  ('mismip'). On a host without glibc, build.sh links
  `tools/oracle_libm/glibc_host.cpp`; its `tanhf` (CORE-MATH's, correctly
  rounded) equals pb's glibc 2.44 `tanhf` on all 2^32 inputs.
- Driver: `run_thermal_solver.F90` builds a one-block domain and calls, as
  `li_time_integration_fe` does:
  1. `li_thermal_init(domain, err)` as on a restart (`config_do_restart`, the
     basal heat flux from file). It copies `config_ice_density` and
     `config_ocean_density` into the module, builds `dsigmaTerm`, and with
     'lapse' sets the surface air temperature.
  2. `li_basal_melt_floating_ice(domain, err)`.
  3. `li_thermal_solver(domain, err)`.
- The stand-ins (`stubs.F90`) resolve the scratch fields the routines
  allocate (`iceCellMask`, `workLevelEdge`) to arrays the driver fills with
  the case's random words. The model leaves them uninitialised, so no output
  may depend on them.

## Coverage

| case | nCells | levels | velocity solver | thermal solver | air temperature | floating melt | calculate bmb |
|---|---|---|---|---|---|---|---|
| `temperature_sia_constant` | 120 | 10 | sia | temperature | file | constant | yes |
| `enthalpy_sia_mismip` | 120 | 10 | sia | enthalpy | lapse | mismip | yes |
| `temperature_none_seroussi` | 96 | 8 | none | temperature | lapse | seroussi | no |
| `enthalpy_none_file` | 80 | 6 | none | enthalpy | file | file | yes |
| `no_thermal` | 64 | 5 | sia | none | file | none | yes |
| `no_thermal_mismip` | 4096 | 3 | none | none | file | mismip | no |

The time step is 30 days. Each case covers:

- halo cells past `nCellsSolve`;
- floating, grounded and thermally inactive columns (thinner than
  `config_thermal_thickness`), and open ocean;
- each boundary row at a grounded bed: melting (basal temperature at the
  pressure melting point), frozen, and, for enthalpy, a temperate bottom
  layer;
- internal melt (temperature) and drainage of water fractions above 0.01
  (enthalpy);
- basal melt and freeze-on, with the basal temperature reset under a dry bed;
- melt under floating ice inside and outside the 'constant' x limit, both
  sides of MISMIP's draft scale, and Seroussi's 50 m/yr cap;
- air temperatures on both sides of 273.15 K.

`velocity solver` 'none' leaves the heat dissipation as read, so those cases
solve with the input's dissipation. The dissipation's zero row for an edge
without two cells is not reached: `li_calculate_mask` reads both cells of
every edge, so the meshes keep them inside.

## RKIND 8

`oracle-r8.npz` holds the same routines compiled without `-DSINGLE_PRECISION`
(RKIND = 8, the same flags otherwise, no FMA instruction), on pb, 2026-11-10.
Each case is drawn from the same seed in float64 instead of rounded to
float32. Its input digests are the `<case>/r8` lines of `inputs.sha256`.
The double `pow` is glibc 2.44's; the port's is `glibc_pow`
(`woof/core/kernels/glibc_flt64.cuh`, MIT). The double `tanh` ('mismip') is
glibc 2.44's, which is CORE-MATH's correctly rounded tanh; the port's is
`g64_tanh::cr_tanh` (`woof/core/kernels/cr_tanh_flt64.cuh`, MIT), equal to
pb's glibc `tanh` on 2^33 CPU and 2^28 GPU arguments
(`tools/oracle_libm/grade_tanh.py`). `enthalpy_sia_mismip/r8` was rewritten
on pb, 2026-11-10, when the RKIND 8 'mismip' refusal was lifted; the other
cases' bytes are unchanged. `no_thermal_mismip` (both precisions) was added then: it
grades MISMIP's `tanh` on about a thousand floating cells. The port refuses 'seroussi' (binary64 `sin`,
glibc's is LGPL and not correctly rounded) at RKIND 8, so
`temperature_none_seroussi` runs 'none' as `temperature_none_none`.
Under the opt-in glibc 2.39 transcription (`WOOF_GLIBC239_LIBM64=1`,
`tools/glibc239_libm_gpu`, LGPL, source checkout only) the port runs
'seroussi' at RKIND 8: `oracle-r8-glibc239.npz` holds `temperature_none_seroussi`
as written, at RKIND 8 (input digest `temperature_none_seroussi/r8-glibc239`),
written on pb 2026-11-12 with the other files' bytes unchanged. The case
evaluates `sin` at one argument, so the words of the transcription itself are
graded by `tools/glibc239_libm_gpu/grade_trig.py` (`verification.log`).
The GPU port runs the `MPAS_RKIND8` unit and must match every word.

## Files

| file | content |
|---|---|
| `oracle-r4.npz` | per case, `<case>/<field>`: `errInit`, `errFloat`, `errSolver`, `logErrors` (all 0), `cellMask`, `vertexMask`, `edgeMask`, `temperature`, `waterfrac`, `enthalpy`, `surfaceTemperature`, `basalTemperature`, `surfaceAirTemperature`, `surfaceConductiveFlux`, `basalConductiveFlux`, `heatDissipation`, `basalFrictionFlux`, `groundedBasalMassBal`, `floatingBasalMassBal` |
| `oracle-r8.npz` | the same fields at RKIND 8 |
| `oracle-r8-glibc239.npz` | the same fields at RKIND 8 for the 'seroussi' case, run under the opt-in glibc 2.39 `sin` |
| `inputs.sha256` | the sha256 of each case's input stream, at each precision (`woof.verify.mali_thermal_solver_oracle.input_bytes`) |

## Grading

`tests/test_mali_thermal_solver_parity.py` and `woof hex ice verify` run
`woof.hex.mpas_ice.thermal_solver_v842` on each case. Every word must be
bit-identical.
