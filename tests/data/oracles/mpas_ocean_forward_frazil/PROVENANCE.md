# MPAS-Ocean v8.4.2 forward mode with frazil oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece forward_frazil` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-20.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- The binary is the forward oracle's (`run_forward.F90`; see
  `tests/data/oracles/mpas_ocean_forward/PROVENANCE.md`): per step the absorbed
  fractions, `ocn_frazil_forcing_build_arrays` (`mpas_ocn_frazil_forcing.F`
  verbatim), `ocn_timestep` and the time-level shift.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`.

## Cases

The forward oracle's culled hexagonal tori with `config_use_frazil_ice_formation`
on, the temperature redrawn from -2.4 to 0.8 C (a sixth of the columns warm):

| case | torus | nVertLevels | steps | what it covers |
|---|---|---|---|---|
| `frazil_split_three_steps` | 7 x 6 | 8 | 3 | the Registry frazil defaults, an odd number of shifts |
| `frazil_land_ice_nested_halos` | 12 x 6 | 8 | 3 | `config_land_ice_flux_mode` 'pressure_only' with a land-ice mask; one block of a decomposition |
| `frazil_limits_no_pressure` | 6 x 5 | 6 | 2 | the surface pressure off, a 0.01 thickness limit, a 60 m cutoff, an even number of shifts |

## Grading

`tests/test_mpas_ocean_forward_frazil_parity.py` and `woof hex ocean verify`
run `woof.hex.mpas_ocean.forward_v842`, and every word must be bit-identical.
