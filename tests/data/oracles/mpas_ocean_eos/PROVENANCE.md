# MPAS-Ocean v8.4.2 equation-of-state oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece eos` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-12.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_equation_of_state{,_linear,_jm}.F` and
  `mpas_ocn_constants.F` (and the other ocean oracles' files), and
  `src/framework/mpas_kind_types.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Reference of record: `-O0 -ffp-contract=off`, built twice:
  - `-DSINGLE_PRECISION` (RKIND = 4, MPAS's default build) gives `oracle-r4.npz`;
  - RKIND = 8 gives `oracle-r8.npz`.

  build.sh fails if an object contains an FMA instruction. The only libm call
  is JM's `exp` in the reference pressure: the host's glibc 2.44 `expf`
  (RKIND 4) and `exp` (RKIND 8). The equation-of-state modules are built in a
  directory of their own, so their `ocn_equation_of_state` module replaces
  `stubs_tendency.F90`'s stand-in there.
- Driver: `run_eos.F90`.
  1. It puts `maxLevelCell` and `refBottomDepth` in the mesh pool, the active
     tracers (one time level) in the state pool's `tracers` subpool with
     `index_temperature`/`index_salinity`, `tracersSurfaceValue` in the
     diagnostics pool and the configs in `ocnConfigs`.
  2. It calls `ocn_constants_init` and `ocn_equation_of_state_init`
     (`config_eos_type` 'linear' or 'jm').
  3. It calls `ocn_equation_of_state_density` as MPAS-Ocean does
     (`ocn_diagnostic_solve`, `ocn_vmix_coefs_rich`), each call into arrays of
     its own that start from the case's sentinel words:
     - `k_displaced` 0, 'relative', with both coefficients: `density`,
       `thermalExpansionCoeff`, `salineContractionCoeff`;
     - 0, 'relative', density only: `densityNoCoeffs`;
     - 1, 'absolute': `potentialDensity`;
     - 1, 'relative': `displacedDensity`;
     - 0, 'surfaceDisplaced', with both coefficients:
       `densitySurfaceDisplaced`, `thermalExpansionCoeffSurface`,
       `salineContractionCoeffSurface`;
     - `nVertLevels`, 'absolute': `densityDeepest`.
  4. It calls `ocn_freezing_temperature` and
     `ocn_freezing_temperature_salinity_deriv` on every (salinity, pressure)
     pair of the case, open ocean and land-ice cavity:
     `freezingTemperature{Open,Cavity}`,
     `freezingTemperatureSalinityDeriv{Open,Cavity}`.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | `<case>/<field>`, every field `(nCells, nVertLevels)` |
| `inputs.sha256` | the input-stream digest of each case at each precision (`woof.verify.mpas_ocean_eos_oracle.input_bytes`) |

The inputs are regenerated, not committed. They use legacy `RandomState`,
reals drawn in float64 and rounded once to float32 for r4.

## Cases

| case | nCells | nVertLevels | EOS | what it covers |
|---|---|---|---|---|
| `linear_defaults` | 48 | 24 | linear | the Registry defaults |
| `linear_configured` | 40 | 30 | linear | other coefficients; open-ocean freezing with every coefficient and a reference pressure |
| `jm_ocean` | 64 | 40 | jm | ocean T/S |
| `jm_clamped` | 48 | 30 | jm | T and S past JM's clamps on both sides |
| `jm_one_level` | 32 | 1 | jm | `k + 1` and `nVertLevels` fall back to level 1 |
| `jm_deep_short_columns` | 40 | 60 | jm | 6 km columns, eight thin (0.05-2 m) surface layers |

`maxLevelCell` runs from 1 to `nVertLevels`: below it the Fortran writes
nothing, and the sentinel words stay. Some freezing pressures sit below, at
and above each reference pressure.

## Grading

`tests/test_mpas_ocean_eos_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.eos_v842`, and every word must be bit-identical.
At the thin surface layers `exp(-0.025 depth) - 1` cancels, so the reference
pressure there carries exp's last bits, but a one-ulp change in it leaves
every output word as it is (the pressure is a few hundredths of a bar against
a bulk modulus of 2e4). So the outputs hold the reference pressure itself,
not exp's rounding. The exp transcriptions are graded on their own
(`woof/core/kernels/glibc_flt32.cuh`, `glibc_flt64.cuh`).
