# Buildings and other obstacles

This page describes the fork's opt-in building data: how building heights are
read from GEM-15 and from open footprint datasets, how they are reduced to the
model grid, and what the result is used for. Nothing here is a WRF default and
nothing runs unless you ask for it.

## What the product holds

`woof.static.buildings` writes one NetCDF per model domain
(`schema = "woof-buildings.v1"`), on the model's mass grid:

| variable | meaning |
|---|---|
| `lambda_p` | building plan area fraction |
| `mean_height`, `std_height` | plan-area-weighted mean and standard deviation of building height (m) |
| `lambda_b` | building surface area (roofs plus walls) per plan area |
| `frontal_x`, `frontal_y` | frontal area index for flow along the model x and y axes (windward wall area per plan area) |
| `frontal_d45`, `frontal_d135` | the same for flow along the diagonals +x+y and +x-y, projected from each wall's own orientation |
| `bldg_hist(bin)`, `bldg_hbin(bin)` | plan area fraction of buildings in each height bin, and their mean height; bins are 2 m to 20 m, 5 m to 100 m, 10 m to 300 m, then 300-400 m (`bin_edges`) |
| `hi(15)` | fraction of building plan area in BEP's 5 m height classes (5, 10, ... 75 m; nearest class) |
| `URB_PARAM(132)` | the NUDAPT block WRF 4.7.1's real init reads |
| `sampled_area` | known (non-hole) ground under each cell, m2 |

`URB_PARAM` is filled at exactly the indices `module_initialize_real.F`
(lines 3182-3215) reads: 91 `LP_URB2D`, 92 `MH_URB2D`, 93 `STDH_URB2D`,
94 `HGT_URB2D`, 95 `LB_URB2D`, 96-99 `LF_URB2D` (the wind sectors of
`module_sf_urban.F`, chosen by `atan2(U10, V10)`: 1 north-south flow,
2 flow along +x+y, 3 east-west flow, 4 flow along +x-y) and 118-132
`HI_URB2D`. Every other index is zero. A raster carries no
building identity, so `MH_URB2D` is the plan-area-weighted mean, the same as
`HGT_URB2D`; NUDAPT's index 92 is a per-building mean. The fork's urban
schemes do not read `URB_PARAM` yet (they use the URBPARM table's
morphology); that ingest is a later change.

The file's attributes carry the sources used, their licences and their
attribution strings, and a receipt of the filters and counts.

## Inputs

**GEM-15 heights.** Building height is GEM-15's surface model minus its
terrain model (`dsm - dtm`, 15 m; the windows come from the existing
`gem15-dsm` / `gem15-dtm` range reader, `woof.static.gem15`).
`ndsm_heights` keeps heights between 2.5 m and 400 m and then applies the
optional filters below. Trees are the main thing these filters try to remove:

- a canopy-spread test: a pixel whose within-pixel spread (for example
  GEM-15's `dsm_percentile_95 - dsm_percentile_5`) is more than half its
  height is rejected, because roofs are flat and tree crowns are not;
- a footprint mask: only pixels inside known footprints are kept;
- a minimum object area (200 m2 by default; one 15 m pixel passes).

GEM-15 is an experimental preprint dataset. Its own paper says buildings
taller than about 50 m are poorly captured, and its regional offsets of
several metres are not corrected.

**Footprints.** `read_footprints` reads a GeoJSON FeatureCollection or a
GeoParquet file with WKB geometry. It decodes the WKB itself, so it needs no
shapely or GEOS. A building's height is taken from the first of these that
exists:

1. its `height` attribute;
2. the median GEM-15 nDSM height over its pixels;
3. its storey count (`num_floors`, `building:levels` or `levels`) times
   3 m;
4. a default height, if you give one.

A building with none of these is dropped and counted in the receipt. Where
footprints overlap, the taller one wins, whatever their order; each
footprint's nDSM median is taken over its own pixels, including those
another footprint covers. When an nDSM median supplies a height, the nDSM's
source (GEM-15) is recorded with its licence and attribution. A GeoParquet
file with no `crs` key is in OGC:CRS84, as the GeoParquet spec says; an
explicit `"crs": null` (unknown) is refused, and a PROJJSON definition
without an authority id is kept whole. woof does not download footprints:
export the area you need with the provider's own tools (for example the
`overturemaps` CLI or an OSM extract) and pass the file.

**Combining.** `aggregate` takes any number of height rasters. Rasters on
one pixel lattice (same CRS, pixel size and alignment, such as adjacent
GEM-15 windows or a footprint burn on a GEM-15 window's grid) are mosaicked
first: where two know the same pixel the taller wins, and walls across the
tile seams are counted once. Rasters on different lattices must not overlap
on the model grid: `aggregate` refuses, because that ground would count
twice. It then supersamples each raster until each model cell holds at
least nine samples, and assigns each pixel to a model cell by its centre. Walls (height steps between
neighbouring pixels) go to a cell by their midpoint. A WRF domain is read
through its own projection (`ProjectionGrid`); idealised cases use the model
plane (`CartesianGrid`, `block_raster`). Holes (NaN) count as neither
building nor open ground.

## Licences

| source | licence | attribution woof records |
|---|---|---|
| GEM-15 v1 nDSM | CC-BY-4.0 | "This dataset is produced by Google." with the Batchu et al. (2026) citation |
| Overture Maps buildings | ODbL-1.0 | "© OpenStreetMap contributors, Overture Maps Foundation" |
| Microsoft GlobalMLBuildingFootprints | ODbL-1.0 | "Microsoft GlobalMLBuildingFootprints" |
| Google Open Buildings | CC-BY-4.0 (v3 polygons also ODbL-1.0) | "Google Open Buildings" |
| OpenStreetMap | ODbL-1.0 | "© OpenStreetMap contributors" |

woof redistributes none of this data. A model input you derive from ODbL
footprints is a Produced Work and must carry the attribution. A derived
database you share publicly must itself be under ODbL.

## Limitations

- Building heats, shading and moisture are not represented. The product
  describes geometry only.
- With pixels assigned by centre, a cell smaller than about three source
  pixels gets the source's blockiness. GEM-15 at 15 m cannot resolve a 10 m
  street.
- Rotation from the source's UTM grid onto the model axes is taken once,
  at the raster centre.
