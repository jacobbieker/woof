# Geostationary imagers for data assimilation (opt-in)

woof can read geostationary imager brightness temperatures and reflectances
so that later stages can assimilate them into initial conditions. The plan,
with its sources and gates, is section 11 of
[dev/WRFDA-3DVAR-PLAN.md](dev/WRFDA-3DVAR-PLAN.md). This page covers what
exists today: the SatWeave reader.

## SatWeave

[SatWeave](https://developers.google.com/weathernext/guides/satweave) is
Google DeepMind and Google Research's global mosaic of GOES-East, GOES-West,
Meteosat 0°, Meteosat IODC and Himawari. It has 17 bands on a 0.1° grid:

- 30-minute steps for 2016-2025;
- hourly steps from 2026, available about an hour after the hour.

```
woof fetch satweave --time 2026-11-18T12:00Z --bbox -105,25,-80,45
```

`woof obs satweave fetch` is the same door.

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

## Not yet

These come next, in the plan's order:

- the native GOES ABI and Himawari AHI readers;
- the opt-in CRTM forward operator;
- the LETKF radiance adapter.

Until then these scenes do not reach an analysis.
