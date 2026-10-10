# High-resolution static geography

## Production path (config-driven, worldwide)

The pilot machinery below is now reachable per-case through one declared
TOML block:

```toml
[static.highres]
enabled = true
cache_root = "/data/highres-cache"
on_refuse = "error"          # or "fallback-30s"
landcover_source = "auto"    # or "cglc-modis-lcz", "annual-nlcd"
```

When enabled, every domain of the case replaces terrain, land-use
fractions/index/mask, and top/bottom soil fractions/categories: terrain
from USGS 3DEP 1/3 arc-second tiles inside the United States and
Copernicus DEM GLO-30 elsewhere, land cover from CGLC-MODIS-LCZ (100 m,
2018, 60 S to 78 N) by default or the Annual NLCD year nearest the case
date on request, and soil from SoilGrids v2 250 m -- fetched on demand for
the domain footprint + halo (`woof.static.highres_fetch`), cached under
`cache_root`, whole published artifacts only (complete 1x1-degree terrain
tiles, the one CGLC-MODIS-LCZ GeoTIFF pinned by size, MD5 and SHA-256, the
complete Annual NLCD year bundle), with the SHA-256 of every fetched byte
recorded into a per-domain receipt at fetch time.  Land-cover sources are
rows of `LANDCOVER_SOURCES` (fetch kind, crosswalk into MODIS 21, water
rule, years, licence), so another collection is a row, not a code path.
CGLC-MODIS-LCZ's Local Climate Zones (51 to 61, WRF's numbering since
4.4.2) become the urban category 13, as WRF's Noah and Noah-MP treat them
when no urban canopy scheme runs, so the category count stays 21 and no
table changes.  Monthly climatologies
stay 30s with the pilot's counted donor fill; TMN is recomputed.  Absence
of the block is the identity: the 30-arc-second build runs byte-unchanged.

Every cell a source does not cover (the sea past the land-cover
collection's offshore edge, the far side of a national border, a terrain
tile that is not published, a footprint that runs past the land-cover
raster) takes the 30-arc-second baseline for that field, handed over
across five cells at the edge with the nest terrain ramp
(`COVERAGE_BLEND_CELLS`), and the receipt's `coverage` entry gives the
count and the latitude/longitude bounds of those cells per field; one
console warning names them.  The lane refuses loudly, with the reason in
the receipt, when a requested source reaches no part of the footprint,
when the baseline land-use inventory is not MODIS-21, when a cell is
covered by neither the source nor the baseline, and when an enabled block
that reached cells would replace zero of them.  A coastal footprint is
not refused: CGLC-MODIS-LCZ separates the sea from inland water itself,
and NLCD's single open-water class is split against the domain's own 30s
baseline water field, so the sea stays WRF ocean category 17 and inland
water becomes lake category 21, with the cell counts and the rule in the
receipt.  `on_refuse = "fallback-30s"` proceeds on the unchanged baseline
beneath a receipt that names the refusal; the default stops the case.  A
case far from the land-cover map's year (2018 for CGLC-MODIS-LCZ, before
1985 or after 2024 for NLCD) takes the nearest map and the receipt names
the anachronism in years.

## The pilot

RW-WPS previously used only the standard 30-arc-second WPS geography tree
for its production static fields.  That is roughly 0.8 km east-west in Ohio, so it is
not a genuinely sub-kilometre description even when the model grid is 500 m
or 333 m.  The opt-in code in `woof.static.highres` is the first narrow,
provenance-bound path beyond that baseline.  It reads local GeoTIFF subsets
directly and does not invoke `geogrid.exe`.

This is a pilot, not a claim of global or production support.  The normal
static builder and public RW-WPS command remain unchanged.

## Pilot data pack

The first pack covers an 18 km square around Caesar Creek, Ohio, inside the
real74 d04 footprint.  Every downloaded byte is bound by path, byte count,
SHA-256, source URL, nominal resolution, reference year, and licence in the
pilot manifest.

| Field | Source | Native resolution | Terms | Pilot treatment |
| --- | --- | ---: | --- | --- |
| Bare-earth terrain | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services) | 1/3 arc-second, about 10 m | US public domain | Area-average to the WRF spherical-earth projected cells, then one WPS smooth/desmooth pass |
| Land cover and inland water | [Annual NLCD Collection 1.2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-land-cover-annual-nlcd-collection-12-land-cover) | 30 m | US public domain | Area fractions, explicit NLCD-to-WRF-MODIS-21 crosswalk; open water becomes ocean category 17 where the 30s baseline water field says ocean and lake category 21 elsewhere |
| Sand, silt, clay | [SoilGrids v2](https://docs.isric.org/globaldata/soilgrids/wcs.html) | 250 m | [CC BY 4.0](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html) | Thickness-weighted 0--30 cm and 30--100 cm medians, normalized and classified with the USDA texture triangle |

The crosswalk has one open water class and cannot tell a lake from the sea,
so the ocean/lake split is made against the domain's own 30-arc-second
baseline water field: a cell the baseline calls WRF ocean category 17 keeps
the open water fraction as ocean, and everywhere else it becomes lake
category 21.  Both counts and the method are written into the receipt.  The
discriminator is already on the model grid at 30 arc-seconds, which is finer
than any vendored coastline polygon and needs no extra dependency.

Annual NLCD begins in 1985.  The April 1974 case therefore uses the earliest
available map, which is still an explicit **11-year anachronism**.  It is an
experiment in spatial detail, not a historically exact 1974 surface.  A
modern map must never be presented as a silent historical replacement.

## Required Noah fields

The pilot replaces terrain, land-use fractions/index/mask, and top/bottom
soil fractions/categories.  It retains the hash-bound 30-arc-second WPS
monthly green fraction, LAI, albedo, snow albedo, and deep-soil temperature.
Where the new water mask exposes land that was water in the old mask, the
climatologies use a deterministic nearest old-land donor and the receipt
records the number of affected cells.  Newly identified water gets WRF water
fills and soil category 14.  `TMN` is recomputed from the merged surface.

SoilGrids does not provide a direct WRF soil category.  The pilot records the
raw sand+silt+clay total, normalizes the three components, and applies the
USDA texture rules.  Missing target land cells use the existing WPS soil
fractions only as an explicit, counted fallback.

## Scientific and operational gates

The implementation:

- verifies each source SHA-256 before decoding it;
- uses WPS's spherical Earth and mass-point registration on every WPS
  projection this tree builds (lambert, mercator, polar): the overlay
  resamples through the grid's own PROJ CRS, so no projection is
  singled out;
- performs continuous area averaging and categorical area-fraction
  aggregation rather than nearest-neighbour sampling at model-cell centres;
- reads no network data, and hands every cell its sources do not cover
  to the 30-arc-second baseline, counted and bounded in the receipt;
- emits comparison arrays, plots, timings, source/fallback audits, and a
  receipt that states exactly what was and was not certified.

The pilot does **not** certify full real74 d04 coverage, a forecast
improvement, stock-WRF parity, historical surface fidelity,
or a public high-resolution-pack CLI.

## Caesar Creek result

The hash-bound pilot completed at both target spacings.  Timings below are one
Windows workstation run and are not a performance certification.

| Grid | 30s baseline build | High-res overrides | Terrain RMSE / max difference | Changed water mask | Changed land-use class | Changed top-soil class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 x 36 at 500 m | 0.188 s | 0.788 s | 3.591 / 21.832 m | 34 / 1,296 | 900 / 1,296 | 671 / 1,296 |
| 54 x 54 at 333.333 m | 0.206 s | 0.472 s | 4.816 / 36.058 m | 89 / 2,916 | 2,031 / 2,916 | 1,520 / 2,916 |

At 500 m, the new mask exposed 13 land cells and masked 21 old land cells;
the SoilGrids subset covered every target land cell.  At 333 m, those counts
were 38 and 51, with three land cells per soil layer using the declared WPS
fallback.  The large class-change counts are expected when comparing a 30 m
NLCD crosswalk against the older 30-arc-second MODIS/USGS products, but they
are not by themselves evidence that the new categories are more accurate.

## Global sources

The global sources are wired and default outside the United States:
[Copernicus DEM GLO-30](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html)
for 30 m terrain,
[CGLC-MODIS-LCZ](https://doi.org/10.5281/zenodo.7670653) for 100 m land
cover (the default everywhere, CC BY 4.0), and SoilGrids 250 m for soil
texture (CC BY 4.0).  Copernicus GLO-30 is available under its Full, Free
and Open licence.  CGLC-MODIS-LCZ represents 2018, so a case far from
2018 carries a named anachronism in its receipt.

Coastline and lake separation follows each land-cover source's water rule.
For a source with one open-water class the split runs against the
domain's own 30-arc-second baseline water field.  It is not optional and
there is no flag for it.  The mask is a required argument of
`build_highres_overrides`, and the production overlay and the bounded
pilot below both derive it from
`woof.static.highres.baseline_ocean_mask`, so the two cannot report
different coastlines for one domain.

## Hex statics (MPAS meshes)

A hex static comes from `rw_mpas_static`, which reads WPS_GEOG at 30
arc-seconds (about 900 m).  For 50-100 m cells, post-process it from the
same cached sources with `woof hex static-highres`.  Give it the input as
`--static S` and the new file as `-o S2`.  `--terrain glo30` is the
default; add `--landuse cglc` for land use and `--offline` to stay off the
network.

- **Order.**  Run it before the vertical grid is built (the vertical build
  smooths `ter`) and before the mesh row is registered (the registry pins
  the static's SHA-256).  A static that already carries a vertical grid,
  or that was already post-processed, is refused.
- **Terrain.**  Copernicus GLO-30 is area-averaged over each cell's
  Voronoi polygon into `ter`.  The tiles are fetched, cached and hashed by
  the same code as the WRF path, under the same per-user cache.  Area an
  unpublished (all-water) tile leaves uncovered takes the cell's own
  30-arc-second height.
- **Land use** (`--landuse cglc`).  `ivgtyp`/`lu_index` become the
  per-cell area mode of CGLC-MODIS-LCZ, through the WRF path's crosswalk.
  A cell with under half its area classified keeps its old category.
  `landmask` is re-derived as `ivgtyp != iswater_lu`, as `rw_mpas_static`
  derives it.  A cell that becomes land takes its land-only climatologies
  and soil categories from the nearest old land cell.  A cell that becomes
  water gets the static's water zeros.
- **Kept on purpose.**  `var2d`, `con`, `oa1..4` and `ol1..4` stay as
  they are.  They are orographic drag statistics, defined over a
  30-arc-second box sized from `dcEdge`.
- **Which cells.**  Only cells finer than `--max-spacing-km` (default 2)
  are processed, so a global variable-resolution parent is sampled over
  its refined region only.
- **Method.**  The rasters are decoded and warped by the Rust
  `static-fields` library onto a Mercator sample lattice.  Each sample goes
  to the cell with the nearest centre, which is the Voronoi assignment, and
  at the edge of the processed set each sample is also tested against the
  cell's polygon.  The per-cell sums are area-weighted.
- **Offline.**  `--offline` refuses, naming the files, when a tile or the
  land-cover raster is not already cached.
- **Provenance.**  The receipt (`<S2>.static-highres.json`) and the
  output's `highres_*` global attributes record the tile hashes, the window
  hash, the lattice and the method.

## Reproducing the bounded pilot

The raster work runs in the Rust `static-fields` library, which a plain
install stages, so nothing extra is needed; `pip install 'recast-woof[geog]'`
adds rasterio and pyproj, the pure-Python parity fallback behind
`WOOF_STATIC_PYTHON=1`. Run:

```text
python -m pip install -e "."
python tools/run_highres_geog_pilot.py \
  --manifest /path/to/pilot-manifest.json \
  --geog-root /path/to/WPS_GEOG \
  --output /new/output/directory
```

The command refuses to overwrite an existing result directory.  Its receipt
binds the source rasters, the WPS baseline index files, code commit, generated
arrays, plots, timings, and quantitative comparisons.
