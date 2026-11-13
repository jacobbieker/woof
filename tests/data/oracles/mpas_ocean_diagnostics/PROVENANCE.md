# MPAS-Ocean v8.4.2 ocn_diagnostic_solve oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece diagnostics` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13. The same build on the mac (arm64, Homebrew gfortran, the
repository's glibc transcriptions for `expf`/`exp`) writes the same words.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_diagnostics.F`, `mpas_ocn_diagnostics_routines.F`,
  `mpas_ocn_equation_of_state{,_linear,_jm}.F` and `mpas_ocn_constants.F` (and
  the other ocean oracles' files), `src/framework/mpas_constants.F` and
  `mpas_kind_types.F`.
- `ocn_diagnostic_solve`, `ocn_diagnostics_init` and
  `ocn_compute_land_ice_flux_input_fields` are cut verbatim out of
  `mpas_ocn_diagnostics.F` (awk ranges) into `diagnostics_cut_head.F90`'s
  module, which repeats the module's USE list and variables less what only
  its other routines need (stream manager, IO units, vector reconstruction,
  GM, ALE thickness).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment, `-O0 -ffp-contract=off`, built twice:
  `-DSINGLE_PRECISION` (RKIND = 4) gives `oracle-r4.npz`, RKIND = 8 gives
  `oracle-r8.npz`. build.sh fails if an object contains an FMA instruction.
  The only libm call is JM's `exp` (`expf` at RKIND 4).
- Driver: `run_diagnostics.F90` (its stream is `oracle_stream.F90`'s; the
  input digests were rewritten on 2026-11-13 when pool names widened to 24
  characters, the outputs unchanged), a self-describing one: dt, every config by
  name, every field by pool and name (`mesh`, `state`, `tracers`,
  `diagnostics`, `forcing`, `scratch`). Fields of an inactive package
  (`landIcePressurePKG`, `frazilIce`) are unassociated pointers, as MPAS
  leaves them. It calls `ocn_constants_init`, `ocn_equation_of_state_init`,
  `ocn_diagnostics_init` and `ocn_diagnostic_solve(timeLevelIn=1)`, then
  writes every real and integer field back in the input's order.
- The stubs (`stubs.F90`) fill a scratch field with `huge()`, so an output
  that read a scratch word before it was written would show it. They take
  one pool slot per put (gfortran -O0 evaluated `slot()` twice inside a
  pointer assignment's subscript) and hold 192 entries.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | `<case>/<pool>/<field>` for every real and integer field the block carries, after the call |
| `inputs.sha256` | the input-stream digest of each case at each precision (`woof.verify.mpas_ocean_diagnostics_oracle.input_bytes`) |

Every cell, edge and vertex field carries MPAS's missing-neighbour element
(`nCells+1` ...). The inputs are regenerated, not committed.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `zmid_linear` | 48 | 16 | 'pressure_and_zmid', linear EOS, one halo extent |
| `jts_jm_apvm_kpp_land_ice` | 56 | 24 | 'Jacobian_from_TS' (in-situ coefficients), JM, APVM, KPP, vertex KE, land-ice pressure ('pressure_only'), frazil, nested halos |
| `montgomery_jm_rk4_kpp_deep` | 40 | 20 | 'MontgomeryPotential', RK4, vertex KE, a KPP layer deeper than the columns |
| `jdens_linear_apvm` | 44 | 30 | 'Jacobian_from_density', APVM |
| `one_level_kpp` | 32 | 1 | one level, KPP, nested halos |

The meshes are random connectivity in which about one edge in twelve and one
vertex corner in fifteen point at the missing cell. They are built so that
the routine reads no scratch word it has not written. `maxLevelVertexBot`
covers every level read at a vertex, and `normalVelocity` is zero below
`maxLevelEdgeTop`, where v8.4.2's APVM multiplies a normal vorticity gradient
it never computed by it.

## Grading

`tests/test_mpas_ocean_diagnostics_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.diagnostics_v842`, and every word of every field must be
bit-identical, the missing-neighbour elements included.
