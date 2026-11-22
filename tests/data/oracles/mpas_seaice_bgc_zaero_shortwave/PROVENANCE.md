# MPAS-Seaice v8.4.2 column-package oracle: the z-aerosols' shortwave bioabsorption

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases zaero_shortwave` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-22.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build as `../mpas_seaice_bgc_zaero`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers`, `config_use_zaerosols` (alone or with
  `config_use_nonreactive`) and `config_use_shortwave_bioabsorption`, with
  and without `config_use_modal_aerosols`; every other biogeochemistry flag
  off. The step cases add `woof_v842_corrections =
  ('bgc_inactive_diagnostics',)` as in `../mpas_seaice_bgc_zaero_step`.
  `nzAerosols` is 3 and `nBioLayers` 7.
- Driver: `run_column.F90`, stages `radiation` (`column_radiation`: the
  delta-Eddington shortwave), `column_step` and the coupled step with the
  velocity solver (3 steps each). With the bioabsorption:
  - `compute_shortwave_trcr` remaps each aerosol's ice layers onto the
    shortwave grid into `bioTracerShortwave`, with `column_radiation`'s
    `verticalGrid` as the biology grid (`mpas_seaice_column.F:2415`; every
    case has at least as many ice layers as biology layers);
  - `compute_dEdd` folds the aerosols' optics into the snow's and the ice's
    layers, from the aerosols pool's bulk tables or the modal ones.
- Cases: `BGC_ZAERO_SW_CASES`: the radiation with the bulk optics and level
  ponds; with the nonreactive tracer, the modal optics, CESM ponds, BL99, the
  Gregorian calendar and the June solstice's noon; with the modal optics,
  topo ponds and nine ice layers; the column step with the modal optics; the
  coupled step with the nonreactive tracer and the bulk optics, Rothrock
  ridging. `_zaero_shortwave_state` draws their aerosols, optics tables and
  (for the radiation) grids and brine fraction; in the radiation cases cell
  2's first category has a negative brine fraction, so `remap_zbgc` stops
  and, `colpkg_step_radiation` never resetting its `l_stop`, the cell's
  categories all keep their shortwave tracers at 0.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: the delta-Eddington shortwave's `sin`, `cos`, `atan`, `asin`, `exp`
  and `log` are the host's glibc 2.44; on the GPU they need the LGPL opt-in
  `WOOF_GLIBC239_LIBM64=1`.
