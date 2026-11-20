# Geostationary imagers for data assimilation (opt-in)

woof can read geostationary imager brightness temperatures and reflectances
so that later stages can assimilate them into initial conditions. The plan,
with its sources and gates, is section 11 of
[dev/WRFDA-3DVAR-PLAN.md](dev/WRFDA-3DVAR-PLAN.md). This page covers what
exists today: the SatWeave reader and the GOES ABI reader for the `bkr/geo`
Icechunk stores.

## SatWeave

[SatWeave](https://developers.google.com/weathernext/guides/satweave) is
Google DeepMind and Google Research's global mosaic of GOES-East, GOES-West,
Meteosat 0°, Meteosat IODC and Himawari. It has 17 bands on a 0.1° grid:

- 30-minute steps for 2016-2025;
- hourly steps from 2026, available about an hour after the hour.

```
woof obs satweave fetch --time 2026-11-18T12:00Z --bbox -105,25,-80,45
```

`woof fetch satweave` is the same door.

### What `fetch` does

- It reads the bands (default 6200, 6900, 7330 and 11200 nm) for one time,
  or for a span with `--end`, inside the box (`west,south,east,north`; west
  greater than east crosses 180°).
- It caches each subset under `~/.woof/cache/satweave`, or
  `$WOOF_SATWEAVE_CACHE` or `--cache`. Each subset is a `satweave.nc` with a
  `provenance.json` beside it. The provenance records the store, the
  SHA-256 of every chunk read, and the licence and attribution. A cached
  subset whose digest no longer matches is refused, not read.
- It prints one line per time, saying which satellite covered how many
  pixels.

### How the data are read

- **Transport.** Reads are anonymous HTTPS from
  `storage.googleapis.com`, so no Google account or client library is
  needed.
  - Python 3.14 decodes the zstd chunks with its standard library. Older
    Pythons need `numcodecs`.
  - `--root gs://satweave_1` is read through the same public HTTPS
    endpoint.
  - `--root` (or `$WOOF_SATWEAVE_ROOT`) may also name a local directory
    that mirrors the store layout.
- **Cost.** The stores hold one whole-globe chunk per band per step, about
  8 MB each. A small box therefore costs the same download as the whole
  globe. The cache keeps only the box.
- **Times.** Archive times must be on :00 or :30, and 2026-onward times on
  the hour. `--nearest` rounds instead of refusing.
- **NaNs.** These are SatWeave's own: beyond the geostationary disc, for
  bands a sector's satellite does not carry, and for reflectances at night.
  Values outside a band's published `valid_range` are set to NaN and
  counted in the provenance.

### Satellite and view geometry

SatWeave does not publish which satellite saw a pixel, nor the view angles
or the scan time. woof reconstructs them per pixel.

- **The satellite.**
  - Band availability decides first: 6.9 µm is carried by GOES and
    Himawari but not Meteosat.
  - Then the smallest view zenith decides, from each sector's nominal
    sub-satellite longitude.
- **The flags.**
  - Every pixel is marked as reconstructed.
  - Pixels a second satellite sees under 70° are flagged as overlaps,
    because SatWeave blends them.
- **Dates.** The table starts on 2018-11-14, the first date when every
  sector is ABI, AHI or SEVIRI. Earlier scenes are refused for
  reconstruction.
- **GOES-19.** From 2025-04-04 GOES-East is GOES-19. CRTM v2.4.1 has no
  GOES-19 coefficients that it can read, so the scene names GOES-16's
  `abi_g16` as the sensor to simulate it with, and says so.

### Licence

SatWeave is CC BY 4.0. Every scene, cache entry and command run carries the
attribution, and a product built from these data must keep it:

> SatWeave geostationary mosaic by Google DeepMind and Google Research,
> licensed CC BY 4.0. Contains modified EUMETSAT Meteosat data; NOAA GOES-R
> series data via Google Cloud Public Datasets; JMA Himawari-8/9 data via
> NOAA NODD on Google Cloud, modified by Google.

woof commits and ships no SatWeave data. SatWeave is experimental and is
not an official forecast or warning product.

## GOES ABI from the `bkr/geo` Icechunk stores

Source Cooperative's `bkr/geo` holds geostationary imagery as Icechunk
repositories that anyone can read. woof reads GOES ABI from
`bkr/geo/virtualized/`: one repository per satellite and channel
(`goes{16,17,18,19}_radf_C{01..16}`), whose chunks are references into
NOAA's own ABI L1b full-disk files. A read is NOAA's radiance with its
quality flags, calibration and navigation, so the satellite, view angles and
scan time are known rather than reconstructed.

```
woof obs geo fetch --satellite 16 --time 2025-03-01T18:00Z --channels C08,C09,C10,C14 --bbox -105,25,-80,45
woof obs geo list --satellite 19 --channel 8 --coverage
```

`woof fetch geo` is the same door as `woof obs geo fetch`.

- **Values.**
  - Channels 7-16 come out as brightness temperature, converted with the
    file's own Planck constants.
  - Channels 1-6 come out as reflectance factor (%), from `kappa0`.
  - A pixel whose quality flag is not 0 is NaN, and is counted in the
    provenance.
  - The 0.5 and 1 km channels are averaged onto the 2 km grid.
  - `--channels` takes ABI channels (`C08` or `8`) or band wavelengths
    (`6200`).
- **Time.**
  - The scan nearest `--time` is used, within `--tolerance-minutes`
    (default 5). A scan's time is the middle of the scan.
  - Each row's time is placed linearly between the scan's start and end.
  - Each channel's archive is split into dated segments: `_<date>` holds
    scans up to that date, and the undated store holds the newest.
  - woof opens the segment that should hold the case, checks its time
    coordinate, and refuses with each segment's span if none has a scan in
    tolerance.
  - A store being written can have an unreadable newest snapshot. woof
    then reads up to 8 snapshots back, and records which one it used.
- **Cache.**
  - Subsets are cached under `~/.woof/cache/geo`, or `$WOOF_GEO_CACHE` or
    `--cache`, as a `scene.nc` with a `provenance.json`.
  - The provenance records the store, snapshot, scan bounds, rejected
    pixel counts and a SHA-256 of the raw counts read.
  - A changed file is refused.
- **Dependencies.** The reader needs `icechunk` and `zarr`
  (`pip install icechunk zarr`); nothing else in woof imports it.
- **Access.**
  - Reads are anonymous.
  - Only NOAA's public `s3://noaa-*` buckets are allowed as sources of
    referenced chunks.
- **Speed.** A 4-channel box over the central US takes about 40 s cold
  and 2.5 s from the cache.
- **Licence.** NOAA's GOES data are open with no restrictions, and NOAA
  asks for attribution. The command prints it.

Docs for developers: section 11.12 of the plan lists every `bkr/geo` store
and its coverage, including Himawari, GK-2A, Meteosat and MTG, which come
next.

## Clear-sky brightness temperatures from CRTM (opt-in)

`woof.rtm` simulates clear-sky infrared brightness temperatures, and their
Jacobians, with JCSDA CRTM v2.4 or later. CRTM is public domain (CC0). woof
ships no CRTM, no compiled library and no coefficients. You build the
library yourself from a CRTM checkout:

```
git clone --branch v2.4.1-jedi.2 https://github.com/JCSDA/crtm CRTM_SRC
pixi exec -s netcdf-fortran -s gfortran -- python -m woof.rtm.crtm_build --source CRTM_SRC --out CRTM_LIB
```

JCSDA tags the 2.4.1 release `v2.4.1-jedi.N`; woof is tested with
`v2.4.1-jedi.2`. The builder refuses a tree that is not the CC0 CRTM v2.4+. That includes
the CRTM 2.3.0 copy inside WRF, which carries GPL files. The builder writes
`libwoof_crtm` and a `build-receipt.json` with the version, the licence
hash and the library hash.

The coefficients come from CRTM's own coefficient package. Put each
sensor's `<sensor>.SpcCoeff.bin` and `<sensor>.TauCoeff.bin` (ODPS) and the
infrared emissivity files in one directory. Then set `WOOF_CRTM_LIB` (the
library or its directory) and `WOOF_CRTM_COEFFS` (that directory).

- `woof.rtm.profiles.profiles_from_wrf_state` turns woof's prognostic
  columns into CRTM layers: `T`, `P`/`PB`, `QVAPOR`, `PSFC`, `P_TOP`,
  `TSK`, `U10`/`V10`, `LANDMASK`, `XICE`, `SNOWC` and `HGT`. woof carries
  no ozone, so ozone is an approximate US Standard Atmosphere profile
  unless you pass one in.
- `profiles_from_pressure_levels` does the same from an isobaric analysis,
  such as GFS or ERA5.
- `woof.rtm.crtm.ClearSkyOperator(sensors).simulate(...)` returns the
  brightness temperatures. With `jacobian=True` it also returns dBT/dT,
  dBT/dq, dBT/dO3 and dBT/dTskin.
- Each scene's platform names its CRTM sensor: `abi_g16`, `abi_g17`,
  `abi_g18`, `ahi_himawari8`, `ahi_himawari9`, `seviri_m08` to `seviri_m11`.
  GOES-19 uses `abi_g16`, because the `abi_g19` file in 2.4.1 is broken.
- CRTM keeps module state, so only one operator can be open per process.
- Columns viewed at more than 65 degrees sensor zenith are refused.

### Measured first-guess departures

This is one case, run on 2025-03-01 at 18 UTC. The background is the GFS
0.25° analysis, put on 80 terrain-following layers with GFS ozone. The
observations are thinned to about 0.3° and limited to sensor zenith below
65°. Clear sky is decided by two tests: the 11.2 µm brightness temperature
is uniform over the 3x3 neighbourhood, and its O-B is within 3 K. The
second test cuts the tails of the 11.2 µm statistics, which is why the
other bands are the fairer read.

Values are O-B mean / standard deviation in K:

| source and sector | sensor | clear | 6.2 µm | 6.9 µm | 7.3 µm | 11.2 µm |
|---|---|---|---|---|---|---|
| native ABI (`bkr/geo`), GOES-East, 105-80 W 25-45 N | `abi_g16` | 1035 | -0.02 / 0.94 | 0.77 / 0.90 | -0.26 / 0.75 | 0.24 / 1.29 |
| SatWeave, GOES-East, same box | `abi_g16` | 588 | -0.19 / 0.83 | 0.68 / 0.81 | -0.35 / 0.69 | 0.60 / 1.37 |
| SatWeave, GOES-West, 150-125 W 20-45 N | `abi_g18` | 843 | 0.16 / 1.27 | 0.57 / 1.15 | -0.29 / 0.79 | -0.46 / 0.64 |
| SatWeave, Himawari, 120-160 E 10 S-30 N | `ahi_himawari9` | 2970 | 0.03 / 0.91 | 0.89 / 0.84 | -0.13 / 0.70 | -0.23 / 0.78 |
| SatWeave, Meteosat 0°, 10 W-25 E 20-50 N | `seviri_m11` | 2111 | 0.41 / 0.80 | - | 0.09 / 0.97 | 0.25 / 1.40 |
| SatWeave, Meteosat IODC, 40-70 E 0-30 N | `seviri_m09` | 4023 | 0.39 / 1.05 | - | 0.27 / 0.76 | 0.01 / 1.28 |

Over the same box at the same time, SatWeave minus native ABI averages
0.00 K in every band, with a standard deviation of 0.2 to 0.6 K in the
water-vapour bands. So the two GOES-East rows differ in which pixels pass
the screen, not in calibration.

The 6.9 µm band is 0.6 to 0.9 K warm on both ABI and AHI. That is a
background or operator bias, the kind variational bias correction (VarBC)
removes.

## Into the LETKF (opt-in)

`woof.da.obs_radiance.radiance_to_gridded_obs` turns one scene into LETKF
observation batches, one per band. Its inputs are the scene, the
background ensemble and an open CRTM operator. It works in five steps:

1. **Thinning.** It keeps one pixel per `thin` x `thin` block of model
   columns. That pixel must have every band, a sensor zenith below 65°, no
   limb flag and an attributed satellite.
2. **H(x) per member.** It runs CRTM on every member's column, using the
   pixel's own satellite and geometry.
3. **Clear-sky screening.** A column passes three tests:
   - The 11.2 µm brightness temperature is uniform over the pixel's 3x3
     neighbourhood.
   - Its departure from the ensemble-mean H is within 3 K.
   - If you pass the background condensate, the condensate path above the
     level 100 hPa below the band's peak is small.
4. **Bias correction and gross check.** It subtracts a static per-band
   offset (`bias_k`), then drops departures beyond 3 sigma of the combined
   observation and ensemble spread. Variational bias correction is not
   implemented.
5. **Placement.** Each observation goes at one level per column: the
   model level nearest the peak of the ensemble-mean temperature Jacobian.
   The vertical localisation radius stands in for the width of the
   weighting function. This is the same one-level convention the GOES
   cloud-water-path batch uses.

`wrf_columns(members)` builds the CRTM columns from WRF-named member
states. Any other state works through a function `columns(member, jj, ii)`
that returns `Profiles`. The returned provenance records each band's
counts, its departure statistics and its peak pressures.

### Regional impact case

A test case ran on 2026-10-28 at 12Z over the US Great Plains (28-45N,
105-85W). The ensemble was 31 GEFS members (the 06Z run, 6-hour
forecast), interpolated to a 31-level, 0.5 degree grid. The observations
were SatWeave GOES-East water-vapour bands, with `abi_g16` as the proxy
sensor. Of the 340 model columns, 175 passed the 3x3 uniformity test and
167 also passed the window check, so 167 observations per band went in.
The independent check uses the 523 clear pixels that were not assimilated. The run used a static
bias per band (the mean O-B: -0.10, +1.20 and -0.31 K), 2 K errors,
400 km / 6 km localisation and RTPS 0.9.

| Band | Assimilated O-B -> O-A RMS (K) | Independent O-B -> O-A RMS (K) |
|---|---|---|
| 6.2 um | 1.04 -> 0.65 | 1.03 -> 0.92 |
| 6.9 um | 1.01 -> 0.62 | 1.00 -> 0.91 |
| 7.3 um | 0.79 -> 0.59 | 0.79 -> 0.72 |

The run was scored against WIS2 radiosondes (`wis2_upper_air`). Rows with
an O-B above 10 K (temperature) or 15 m/s (wind) were excluded from both
the background and the analysis score. There were no aircraft reports at
that hour. The analysis did not degrade the radiosonde score:

| Variable | Background RMS | Analysis RMS |
|---|---|---|
| T 850-500 hPa | 0.40 K | 0.40 K |
| T 400-200 hPa | 2.41 K | 2.40 K |
| u, all levels | 4.12 m/s | 4.11 m/s |
| v, all levels | 3.41 m/s | 3.41 m/s |

The increments are small, with an RMS of 0.02 K for temperature and
0.07 m/s for wind. This is expected for three upper-tropospheric
water-vapour channels with 2 K errors. The case shows that the operator
is wired up correctly; it does not show forecast skill.

## Not yet

These come next, in the plan's order:

- Himawari AHI, GK-2A AMI and Meteosat SEVIRI/FCI from the other
  `bkr/geo` stores;
- variational bias correction, and the radiance operator for woof's
  3D-Var (plan section 11).
