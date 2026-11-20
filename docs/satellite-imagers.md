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
  - A changed file is refused by the explicit reader; the fetch path
    treats it as a cache miss and re-reads.
  - The cache is keyed by satellite, UTC time, channels, box, tolerance and
    root, not by the store's contents. A hit is served as is; `--refresh`
    (`refresh=True`) re-reads, and the provenance names the snapshot read.
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

## Not yet

These come next, in the plan's order:

- Himawari AHI, GK-2A AMI and Meteosat SEVIRI/FCI from the other
  `bkr/geo` stores;
- the opt-in CRTM forward operator;
- the LETKF radiance adapter.

Until then these scenes do not reach an analysis.
