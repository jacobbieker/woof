# MPAS-Seaice v8.4.2 column-package oracle: delta-Eddington shortwave

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases dedd` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-20.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, two stages:
  - `radiation`: `colpkg_init_thermo` (`heat_capacity`), `colpkg_init_orbit`
    (as `seaice_init_column_shortwave` calls it for `dEdd`), then
    `column_radiation(domain, clock, .false.)` (`colpkg_step_radiation`:
    `run_dEdd`);
  - `init_shortwave`: `colpkg_init_thermo`, then
    `seaice_init_column_shortwave` (the orbit, the radiation with
    `initonly`, then the cell albedos).
  The clock is `config_start_time`: `get_day_of_year` and
  `get_seconds_into_day` read it.
- Cases: `woof.verify.mpas_seaice_column_oracle.DEDD_CASES`, all
  `config_shortwave_type = 'dEdd'` with `config_calc_surface_temperature`
  true, at several dates and times of day:
  - level ponds (the Registry default, mushy; BL99 at noon on the June
    solstice), CESM ponds with a snow transition depth, topo ponds, and no
    ponds (`shortwave_dEdd_set_pond`) with a snow transition depth;
  - negative and positive tuning parameters (the clamped scattering
    coefficients), no algae absorption, other snow grain parameters;
  - BL99 with one layer and three categories, and with twelve layers;
  - zero-layer thermodynamics (no heat capacity), without and with topo
    ponds;
  - `seaice_init_column_shortwave` with level ponds (`initonly`) and with
    topo ponds.

  The owned cells cover the branches (`_radiation_state`, `_dedd_state`):
  a surface at the melting point and surfaces near it (melting snow grains,
  `set_pond`'s ponds), no ice, sub-puny ice, snow just above and just below
  `hs_min`, bare ice over an ice-free category, thick ice under deep snow
  (level ponds with `rp < 0.15`), no sun, ponds shallower than `hpmin`,
  between `hpmin` and `hp0` and deeper, topo lids straddling the critical
  thickness, and longitudes over [0, 2 pi) with one at pi and two above it.
- v8.4.2's own behaviour, kept word for word:
  - `shortwave_dEdd_set_snow`'s `intent(out)` snow fraction is assigned
    only where `hs >= hs_min`; elsewhere it keeps the 0 `run_dEdd` stored in
    the actual argument (gfortran passes it by reference). Strictly the
    dummy is undefined on entry; the port keeps gfortran's behaviour;
  - `shortwave_dEdd` raises the solar zenith cosine to `puny` wherever a
    category has ice and the sun is up, and `column_radiation` writes it
    back to `solarZenithAngleCosine`;
  - without heat capacity `compute_dEdd` folds the first ice and snow
    layers' absorption into the surface flux once per surface type, on the
    sums so far;
  - categories without ice keep their bare-ice, snow and pond albedo words;
    `colpkg_step_radiation` zeroes `bioTracerShortwave`;
  - the chlorophyll absorption without the biogeochemistry is
    `kalg*(0.5/dzk(klev))` in the bottom layer's visible band only.
- Not in the cases: ice without volume (`aice > puny`, `vice = 0`). Its
  layers have no thickness, so `kalg*(0.5/0)` and then `inf*0` make NaN in
  v8.4.2 and in the port alike, but a NaN's payload is the host's, so its
  words cannot be compared. `_dedd_state` gives that category volume.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: `compute_dEdd` and `solution_dEdd` call `exp`, the host glibc
  2.44's; on the GPU that is `glibc_exp` (`glibc_flt64.cuh`), the MIT
  transcription of the same routine. `shr_orb_params` calls `sin`, `cos` and
  `atan`, and `shr_orb_decl` and `compute_coszen` call `sin`, `cos` and
  `asin`, glibc's, which are not correctly rounded: the GPU matches them only
  through the LGPL transcriptions behind `WOOF_GLIBC239_LIBM64=1`
  (`g239_sin`, `g239_cos`, `g239_atan`, `g239_asin`;
  `tools/glibc239_libm_gpu/`, graded against glibc 2.44 by `grade_trig.py`).
  Without the opt-in the delta-Eddington shortwave is refused by name.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/column_dedd_v842.cu` is a modified derivative of the
column package's `ice_shortwave.F90`, `ice_orbital.F90` and `ice_colpkg.F90`
and of `mpas_seaice_column.F`, under the same licence
(`licenses/LICENSE-MPAS-BSD-3-Clause.txt`). The opt-in `g239_asin` is new
LGPL code in `tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh` (see
`NOTICE`), compiled only with `WOOF_GLIBC239_LIBM64=1` and kept out of the
wheels.
