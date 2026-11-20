# MPAS-Seaice v8.4.2 column-package oracle: the stand-alone ocean mixed layer

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases oml` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-20.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. None of the 40 column objects nor
  `run_column_r8` contains an FMA (checked with objdump).
- Driver: `run_column.F90`.
  - Stage `ocean_mixed_layer` calls `seaice_column_ocean_mixed_layer(domain)`
    (`mpas_seaice_column.F:4320-4543`): per owned cell, `colpkg_atm_boundary`
    with sfctype `'ocn'` (`ice_atmo.F90`'s `atmo_boundary_layer`), the four
    ocean albedos set to `seaiceOceanAlbedo`, and `colpkg_ocn_mixed_layer`.
    Then, under `config_use_test_ice_shelf`, the freezing/melting potential
    is zeroed where `landIceMask` is 1.
  - Stage `column_step` (see `../mpas_seaice_column_step/PROVENANCE.md`)
    with `config_use_ocean_mixed_layer`, so that
    `seaice_column_coupling_prep` runs the mixed layer first in each step's
    post-dynamics.
- Pools: the vertical thermodynamics' (`VT_POOLS`; `STEP_POOLS` for the
  steps). `seaice_column_ocean_mixed_layer` reads `nCellsSolve` from the
  `ocean_coupling` pool, so these cases' streams carry it there too.
- Cases: `woof.verify.mpas_seaice_column_oracle.OML_CASES` and
  `OML_STEP_CASES`:
  - the Registry defaults (stresses on, 5 iterations);
  - `config_calc_surface_stresses = .false.`;
  - high-frequency coupling (`umin` 0.5) with one iteration;
  - the test ice shelf with ten iterations;
  - the column step with BL99 and dEdd, three steps;
  - the column step on the Registry defaults (mushy, level ponds) without
    the vertical thermodynamics, two steps.

  The state is `_thermo_state`'s air over an ocean from `_oml_ocean`: a sea
  surface below, at and well above freezing; a shallow mixed layer, whose
  potential the +-1000 W m-2 clamp bounds; a warm-air cell (a stable
  boundary layer); and a positive deep heat flux under a mixed layer at
  freezing.
- v8.4.2's own behaviour, kept word for word:
  - the grid field `airOceanDragCoefficientRatio` is passed as the ocean
    call's `Cdn_atm_ratio_n`, so under `config_calc_surface_stresses` the
    open ocean's ratio replaces the ice's merged one (nothing in v8.4.2
    reads the field afterwards);
  - `oceanHeatFluxConvergence` is zeroed in place where the mixed layer is
    at or below freezing and the flux would cool it;
  - the open ocean's `atmo_boundary_layer` never takes the high-frequency
    wind or stress branches (those are the ice's); only `umin` changes.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object, the abort status and the
  log lines. `inputs.sha256`: the input-stream digests.
- libm: `atmo_boundary_layer`'s `psimhu` calls glibc's `atan`, which is not
  correctly rounded. The GPU matches it only through the LGPL transcription
  behind `WOOF_GLIBC239_LIBM64=1` (`tools/glibc239_libm_gpu/`); without the
  opt-in, the mixed layer is refused by name. `exp` and `log` are
  `glibc_flt64.cuh`'s MIT transcriptions. `TsfK**4` is libgcc's
  `__powidf2`, which gives `(x*x)*(x*x)`.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/ocean_mixed_layer_v842.py` and its kernel in
`column_therm_v842.cu` follow `mpas_seaice_column.F`, `ice_colpkg.F90` and
`ice_atmo.F90`, under the same licence
(`licenses/LICENSE-MPAS-BSD-3-Clause.txt`).
