# MPAS-Seaice v8.4.2 column-package oracle: the vertical biochemistry's algae and nutrients

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases algae` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-22.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build as `../mpas_seaice_bgc_zaero`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers`, `config_use_vertical_biochemistry` and
  `config_use_nitrate`, with and without `config_use_ammonium` and
  `config_use_silicate`, the z-aerosols, the nonreactive tracer and
  `config_use_shortwave_bioabsorption`; one case carries nitrate, ammonium
  and silicate with the nonreactive tracer and no vertical biochemistry.
  Every other biogeochemistry flag off. `nAlgae` is 3 and `nBioLayers` 7.
- Driver: `run_column.F90`, stages `init` (the column package's init:
  `colpkg_init_zbgc`'s tracer layout and parameters, `colpkg_init_bgc`'s
  profiles) and `biogeochemistry` (`column_biogeochemistry`:
  `update_snow_bgc`, `z_biogeochemistry` with `algal_dyn` per biology
  layer, `merge_bgc_fluxes`).
- Cases: `BGC_ALGAE_CASES`: three inits (the algae and nitrate; with
  ammonium, silicate, a transport type per band and frazil fractions; with
  the z-aerosols and the nonreactive tracer on three ice layers) and eight
  column_biogeochemistry cases (the algae and nitrate; with the nutrients
  and distinct carbon to nitrogen ratios; BL99 with nitrification and a
  maximum brine temperature of -1.5; nutrient limited, with
  `config_maximum_loss_fraction` 0.5 and nitrate, silicate and ammonium
  nearly gone in four cells; the bioabsorption; a fast `algal_vel` with a
  half-hour step, a high minimum optical depth and a salinity factor of 0.8;
  the nutrients, z-aerosols and nonreactive tracer on four ice layers and
  three categories; the nutrients without the algae). `_algae_state` draws
  their layers on top of `_nonreactive_state`'s and gives cell 0's last
  category an area below 0.001.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: `algal_dyn`'s `exp` is the host's glibc 2.44, which the port's
  default `glibc_exp` transcribes; no LGPL opt-in is needed.
