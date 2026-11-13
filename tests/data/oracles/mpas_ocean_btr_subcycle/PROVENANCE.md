# MPAS-Ocean v8.4.2 barotropic-stage oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece btr_subcycle` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256` (it now pins
  `mpas_ocn_time_integration_split.F`).
- `ocn_time_integrator_split`'s declarations, its config lookups and its stage 2
  (from `oldBtrSubcycleTime = 1` to `endif ! split_explicit`) are cut verbatim
  (awk ranges) into one routine (`btr_subcycle_cut_head.F90`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object; no libm call.
- Driver: `run_btr_subcycle.F90` (stream: `oracle_stream.F90`). A field's time
  level 2 is the pool entry `NAME@2`. One block, whose halo exchanges are the
  identity, and `nBtrSubcycles` taken from the config of that name. Every
  field is written back.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `defaults` | 48 | 12 | v8.4.2's defaults: gam1 0.5, two velocity correctors, 3 x 2 subcycles, the velocity correction |
| `nested_halos` | 60 | 10 | four nested halo extents: each step over the extent its counter names, the corrector's copy one layer wider |
| `no_predictor_filtered` | 40 | 8 | gam1 0 (no velocity predictor: the SSH predictor reads the new level as it stands), `config_filter_btr_mode`, one corrector, gam2 0.5, no velocity correction |
| `ssh_corrector` | 44 | 9 | `config_btr_solve_SSH2` with no velocity corrector (the cell loop reads the new SSH it is writing), gam1/2/3 0.4/0.7/0.6 |
| `unsplit` | 24 | 6 | `config_time_integrator = 'unsplit_explicit'` |

The meshes are random connectivity on one block. In `nested_halos` each
layer's `edgesOnEdge` lie within the next layer.

## Grading

`tests/test_mpas_ocean_btr_subcycle_parity.py` and `woof hex ocean verify`
run `woof.hex.mpas_ocean.btr_subcycle_v842`, and every word must be
bit-identical.
