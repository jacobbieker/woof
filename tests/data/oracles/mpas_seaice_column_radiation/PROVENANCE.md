# MPAS-Seaice v8.4.2 column-package oracle: shortwave (no delta-Eddington)

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases radiation` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-15.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, three stages:
  - `prep_radiation`: `column_prep_radiation` (the scale of the absorbed
    shortwave by the new incoming shortwave);
  - `radiation`: `colpkg_init_thermo` (`heat_capacity`), then
    `column_radiation(domain, clock, .false.)` (`colpkg_step_radiation`);
  - `init_shortwave`: `colpkg_init_thermo`, then
    `seaice_init_column_shortwave` (the radiation, then the cell albedos).
  The radiation stages gate each Registry var by its own packages only, as
  the Registry allocates them, so `bioTracerShortwave` exists.
- Cases: `woof.verify.mpas_seaice_column_oracle.RADIATION_CASES`:
  - `ccsm3` with constant albedos (BL99; mushy with five layers; zero-layer
    thermodynamics with three categories and the linear-salt freezing
    temperature);
  - `ccsm3` with `compute_albedos` (BL99; on the `new` bounds with a
    thickness limit of 0.7, other band albedos and two categories);
  - coupled runs (`config_use_surface_temperature` false): `dEdd` with topo
    ponds and `ccsm3` with level ponds (BL99), whose shortwave is the
    coupler's;
  - `column_prep_radiation` with one and with three categories (four
    layers);
  - `seaice_init_column_shortwave` with constant albedos, with
    `compute_albedos` and coupled with topo ponds.

  The owned cells cover the branches (`_radiation_state`): a surface at the
  melting point, no ice, a surface just below it with a topo-pond lid,
  sub-puny ice, no snow over an ice-free category, ice without volume,
  thick ice under deep snow (a sub-puny scale), no sun (`coszen` < 0), and a
  zero scale with no ice in the cell.
- v8.4.2's own behaviour, kept word for word:
  - `shortwave_ccsm3` passes `ncat` as `absorbed_solar`'s `nilyr`
    (`ice_shortwave.F90:236`): the light is spread over `ncat` layers of
    `hi/ncat` and the layers below stay zero. With `ncat > nilyr` v8.4.2
    writes past the array, so the port refuses that by name;
  - `shortwave_ccsm3` sets `coszen` to 0.5, and `colpkg_step_radiation`
    zeroes `bioTracerShortwave`;
  - `compute_albedos`' intent(out) snow albedos keep the caller's ocean
    albedo when there is no snow;
  - `seaice_init_column_shortwave` adds the cell's bare-ice, snow and pond
    albedos and `effectivePondAreaCell` to what was there, without zeroing.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: `absorbed_solar`'s Beer law calls `exp`, the host glibc 2.44's. On
  the GPU that is `glibc_exp` (`glibc_flt64.cuh`), the MIT transcription of
  the same routine. `compute_albedos` calls `atan` (the thickness ramp),
  glibc's, which is not correctly rounded: the GPU matches it only through
  the LGPL transcription behind `WOOF_GLIBC239_LIBM64=1` (`g239_atan`), as in
  `../mpas_seaice_column_therm/`. Without the opt-in, `compute_albedos` is
  refused by name; the cases that need it are `RADIATION_TRIG_CASES`. The
  constant albedos, the coupled runs and `column_prep_radiation` need no
  opt-in. delta-Eddington with `config_use_surface_temperature` (the orbit
  and `coszen`: `sin`, `cos`, `asin`, `atan`) is not ported and refused.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/column_radiation_v842.cu` is a modified derivative of
the column package's `ice_shortwave.F90` and `ice_colpkg.F90` and of
`mpas_seaice_column.F`, under the same licence
(`licenses/LICENSE-MPAS-BSD-3-Clause.txt`). The opt-in `g239_atan` is the
existing LGPL transcription; this stage adds no LGPL code.
