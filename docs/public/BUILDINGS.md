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
without an authority id is kept whole. woof downloads
footprints from one provider only, Overture Maps (`--overture`, below);
for any other source export the area you need with the provider's own tools
(for example an OSM extract) and pass the file.

**Combining.** `aggregate` takes any number of height rasters. Rasters on
one pixel lattice (same CRS, pixel size and alignment, such as adjacent
GEM-15 windows or a footprint burn on a GEM-15 window's grid) are mosaicked
first: where two know the same pixel the taller wins, and walls across the
tile seams are counted once. Rasters on different lattices must lie at
least one model cell apart: `aggregate` refuses an overlap (that ground
would count twice) and a shared seam (the walls along it would be lost),
comparing the model cells each raster's outline reaches, so a rotated
or curved raster is judged by its true shape. A raster off the domain is
ignored. Only rasters that sample some cell, and only footprints that
show in the burn, are credited as sources. An nDSM must sit on the footprint
raster's own lattice, and georeferenced footprints cannot burn onto the
model plane (`crs=None`). It then supersamples each raster until each model cell holds at
least nine samples, and assigns each pixel to a model cell by its centre. Walls (height steps between
neighbouring pixels) go to a cell by their midpoint. A WRF domain is read
through its own projection (`ProjectionGrid`); idealised cases use the model
plane (`CartesianGrid`, `block_raster`). Holes (NaN) count as neither
building nor open ground.

## Building the product for a domain (`woof buildings`)

```bash
woof buildings --namelist namelist.wps --domain 4 \
    --footprints buildings.geojson:osm --output buildings_d04.nc
```

The command reads one domain of a WPS namelist (the innermost by default)
and writes its product. `--footprints PATH[:SOURCE]` names a footprint file
and its source (`osm`, `overture`, `microsoft`, `google-open-buildings`, or
`user`); repeat it for more files. The footprints are read over the domain
plus a two-cell halo and burned on one UTM lattice at a quarter of the grid
spacing, but no coarser than 5 m (`--resolution`). A lattice of more than
150 million pixels is refused; pass a coarser `--resolution` for a large
domain, knowing that footprints much smaller than a pixel are lost or grown
to it. GEM-15 is range-read under the same box into the high-resolution
terrain cache (`--cache`, `--offline`):

- with footprints, GEM-15 only supplies the nDSM median for footprints that
  have no `height` attribute; only footprints become buildings, so trees
  and bridges in the nDSM do not;
- without footprints, the filtered nDSM is the product (objects under
  `--min-area`, 200 m2 by default, are dropped); GEM-15 cells on either side
  of a UTM zone edge are resampled onto one 15 m lattice;
- `--no-gem15` uses the footprints' own heights and storeys only.

**Overture Maps door (`--overture [RELEASE|latest]`).** The command fetches
the Overture `buildings` theme, type `building`, for the domain plus a halo
of four cells and about 100 m, and adds it as an `overture` footprint file
(it combines with `--footprints`). It reads the public release
(`s3://overturemaps-us-west-2/release/<release>/theme=buildings/type=building/`)
anonymously through pyarrow's S3 filesystem, filters on each row's `bbox`
struct so only matching row groups are read, and keeps `id`, the WKB
geometry, `height` and `num_floors` (`sources` is read for the sidecar's
counts and not stored in the GeoParquet). A bare `--overture` or
`latest` takes the newest release under the bucket's `release/` prefix; an
id such as `2026-11-18.0` pins one. The result is a GeoParquet file
(`overture-buildings-<release>-<box hash>.parquet`, OGC:CRS84) with a JSON
sidecar of the same name under `<cache>/overture/` (`--cache`, by default
the high-resolution terrain cache). A second run with the same release and
box reads the cache; an explicit release uses no network, and `latest` lists
the bucket first, falling back to the newest cached release (with a note on
stderr) if the listing fails. `--offline` never touches the
network: it uses the cached file (the newest cached release for `latest`)
and refuses with a message if there is none. The sidecar records the release,
box, S3 path, fetch time, footprint count, the footprints per
`sources[].dataset` and per source licence; the receipt in the product
carries it as `overture`.

Licences: the theme is ODbL 1.0 and woof attributes it "© OpenStreetMap
contributors, Overture Maps Foundation". Individual records can come from
other datasets under their own licences (for example Esri Community Maps
contributors and Google Open Buildings under CC BY 4.0, Microsoft Global ML
Building Footprints under ODbL); read the per-dataset and per-licence counts
in the sidecar and see <https://docs.overturemaps.org/attribution/> before
you publish a product or database derived from it. woof ships no Overture
data. The first fetch of a small box took about five minutes
(it reads the footers of the release's 512 files); later runs use the cache.
This is a known cost, not yet optimised. Without a pyarrow that has S3
support the command refuses and says so. Releases are large; the box keeps the download to the row groups
under it.

The receipt is printed and stored in the product. It gives the box, the
GEM-15 windows, and for each footprint file where every height came from
(attribute, GEM-15 median, storeys, default, or dropped). GEM-15 is named
as a source only when a building whose height it gave shows in the domain.

## Resolved obstacles in the model (`obstacle_opt = 1`)

On an LES or other high-resolution domain (grid spacing up to 500 m;
in practice a few metres) the product can be resolved by the dynamics.
Set it per domain:

```toml
[[domain]]
obstacle_opt = 1
obstacle_file = "buildings_d03.nc"   # made on this domain's mass grid
```

The method is a porous immersed obstacle. It is a hybrid of the volume
penalisation (Brinkman) method of Angot, Bruneau and Fabrie (1999), the
masked topography of PALM, and the drag-porosity buildings of Meso-NH
(Aumond, Masson and Lac 2013):

- The solid fraction of each model layer comes from the building height
  histogram and the base-state level heights above ground. A layer at
  least `obstacle_solid_fraction` full is solid. Every u, v or w face next
  to a solid cell is blocked, the C-grid rule.
- A blocked face's coupled momentum is relaxed toward zero at the rate
  `obstacle_relax_factor / dt`.
- Partly built cells get canopy form drag `Cd a_f |U|`. `a_f` is the
  cell's frontal area index spread over its building volume profile, and
  Cd is `obstacle_drag_coefficient` (Coceal and Belcher 2004).
- Open faces beside or above a wall get log-law wall stress
  `(kappa / ln(d / z0))^2 |U| / spacing` for each wall, with d half the
  spacing and z0 `obstacle_wall_z0`. This is the wall model WRF-IBM uses (Lundquist et al.
  2010, 2012; Bao, Chow and Lundquist 2018).

All three act as one rate `k = a/dt + b|U|` per face. Each RK stage
subtracts `k / (1 + k dt)` times the coupled momentum from the face's
tendency, after the stage's other slow tendencies. That bounded implicit
form is stable for any k. The acoustic steps carry it as part of the held
slow forcing, so no acoustic kernel changes. The geometry is static (from
`phb`), so a restart rebuilds the same coefficients. The checkpoint binds
the buildings product's bytes, so resuming against a changed product is
refused. One fused kernel
(`kernels/obstacle_drag.cu`, CUDA and Metal) works on a sparse list of the
faces that touch an obstacle. Off, or with no buildings, nothing is built
and nothing launches, and the run steps to the same bits as before (checked
in fast and strict numerics).

WRF-IBM's cut-cell reconstruction would follow sloped roofs more
closely. It needs terrain-following interpolation stencils on every
boundary point, which a staircase of blocked faces avoids. At a few metres
the staircase is what PALM and most urban LES use.

**Benchmarks** (`woof.verify.cases.obstacle_canyon` and `obstacle_cube`,
RTX A6000):

| case | measure | woof | reference |
|---|---|---|---|
| 2-D canyon, H/W = 1, 10 cells per H | vortex centre x/W, z/H | 0.55, 0.45 | about 0.5 and 0.5 to 0.6 (Brown et al. 2000; Li, Liu and Leung 2008) |
| | recirculation cells up the canyon | 1 | 1 |
| | floor wind / U(2H), roof wind / U(2H) | -0.09, +0.20 | reverse floor flow, forward roof shear |
| | wind inside the buildings / U(2H) | 1.7e-4 | 0 |
| 2-D canyon, H/W = 2 | recirculation cells up the canyon | 2 | 2 (Li, Liu and Leung 2008; Sini, Anquetin and Mestayer 1996) |
| | primary vortex centre z/H | 0.67 | upper canyon |
| | floor wind / U(2H) | +0.02 | forward, under the lower counter-rotating vortex |
| cube, 8 cells per H, periodic 12H x 6H | reattachment behind the cube / H | 1.70 | about 1.6 (Martinuzzi and Tropea 1993), 1.7 (Shah and Ferziger 1997) |
| | horseshoe separation ahead / H | 0.5 | about 1.0 |
| | wind inside the cube / U(2H) | 0.009 | 0 |

The benchmarks run without a surface layer, so the ground takes the LES
closure's prescribed drag (`isfflx = 0`, `tke_drag_coefficient`). They set
it to the walls' own log law (z0 = 1 cm at the first level). With a
free-slip floor instead, no boundary layer approaches the cube: the
horseshoe shrinks to 0.25 H and the wake reattaches at 2.4 H. A real case
runs a surface layer on the ground. Eight cells per H under-resolve the
horseshoe; Martinuzzi and Tropea's 1.0 H is a channel flow at
Re_H = 4e4.

## Terrain under buildings

Resolved buildings (and an urban canopy under LES) sit on top of the
terrain, so the terrain must be bare earth. Copernicus GLO-30, SRTM and the
GEM-15 `dsm` are surface models: over a city their heights include the
roofs, and the buildings would count twice.

A domain that sets `obstacle_opt > 0`, or runs `sf_urban_physics > 0`
(a `[shared]` key) with `km_opt` 2, 3 or 5, needs bare earth. For that
domain only:

- under the sub-km default (no `[static.highres]` block, GLO-30) or
  `terrain_source = "auto"`, it takes the bare-earth USGS 3DEP 1/3" DEM
  where the 3DEP envelope contains its footprint;
- outside 3DEP it is refused (`surface-model-under-buildings`); declare
  `terrain_source = "gem15-dtm"` (bare earth, experimental). Where GEM-15
  would fall back to GLO-30, that is refused too;
- a surface model you name yourself is kept, with one warning per domain.

The other domains of the tree, and configurations without buildings,
resolve exactly as before. The building domains are part of the static
identity (`bare_earth_domains`), so a preparation made without them is not
reused.

## A city check: Manhattan at 50 m and 25 m

This check asks whether individual buildings show up in a real-data LES, and
whether their heights in the model match the source data.

**Setup.**
- `woof domain` built a nested HRRR-route tree from 3 km down to 50 m:
  - 3 km, 1 km and 250 m parents, then a 50 m domain (370 x 320 cells,
    18.5 x 16 km) over Manhattan, Brooklyn and Queens;
  - a one-way 25 m nest (120 x 300 cells, 3 x 7.5 km) from the Battery to
    59th St;
  - 112 levels, about 5 m apart at the ground and 10 m apart at 100 m.
- Physics: the 250 m, 50 m and 25 m domains run LES (`km_opt = 3`, PBL off).
- Terrain: the bare-earth 3DEP 1/3" DEM (`[static.highres] terrain_source =
  "usgs-3dep-13as"`). On a city, use a bare-earth source. Copernicus
  GLO-30, the sub-km default, is a surface model with the buildings
  already in it. Since phase 2 a configuration with `obstacle_opt` (or an
  urban canopy under LES) does this itself; see "Terrain under buildings".
- Buildings: `woof buildings` made both products from an OpenStreetMap
  extract (256,305 footprints in the 50 m domain), with GEM-15 supplying
  missing heights. The 50 m product used a 12.5 m lattice, the default
  (dx/4) before #585's review fix capped it at 5 m. The 25 m product used
  a 5 m lattice.
- Run: HRRR initial and boundary data from 2026-11-02 19 UTC, wind about
  4 m/s from the SSW, on an RTX A6000 that other jobs were also using.
- Comparison: one run with buildings (`obstacle_opt = 1` on the 50 m and
  25 m domains) against one without. Fields are 10-minute output averaged
  over 19:20-20:00, so the first 20 minutes of spin-up are left out.

**Source heights.** GEM-15's median surface height over each footprint
(nDSM, surface minus terrain), compared with the OSM `height` tag (220,782
footprints):

| OSM height | footprints | GEM-15 minus OSM (bias) | RMSE | r |
|---|---|---|---|---|
| all | 220,782 | -3.1 m (-20%) | 8.5 m | 0.65 |
| 0-10 m | 133,563 | -1.5 m (-16%) | 2.9 m | 0.43 |
| 10-20 m | 72,963 | -3.2 m (-24%) | 4.9 m | 0.49 |
| 20-50 m | 11,718 | -10 m (-34%) | 12.7 m | 0.53 |
| 50-100 m | 2,138 | -35 m (-53%) | 39 m | 0.03 |
| 100-200 m | 348 | -107 m (-82%) | 112 m | -0.09 |
| 200-600 m | 52 | -253 m (-92%) | 263 m | 0.03 |

GEM-15 is good for low-rise buildings but loses towers. Its 15 m pixels
blend roofs with streets, and 21-26% of the pixels under towers over 100 m
are holes. Use it to fill missing heights under footprints, which is what
`woof buildings` does. GEM-15 on its own is a poor building map:

- it marks 85,714 cells built where OSM marks 48,656, because trees,
  bridges and highways pass its filters;
- its cell mean height has an 18.8 m RMSE against OSM.

OSM footprints with GEM-15 filling the gaps match the footprint-only
product to within +0.06 in plan area fraction and +0.7 m in mean height.

**Heights in the model.** The model's building height is the top of its
solid cells at each footprint centre, on the run's own levels. This
compares it with the OSM tag for footprints whose bounding box is at least
2,500 m2:

| domain | footprints | with a solid column | bias | median | RMSE | r |
|---|---|---|---|---|---|---|
| 50 m | 4,315 | 2,478 | -8.0 m | -2.3 m | 25 m | 0.86 |
| 25 m | 841 | 793 | -1.8 m | -1.0 m | 7.0 m | 0.99 |

At 25 m the model's buildings stand at their source heights, to within
about one level spacing. At 50 m, 43% of these footprints have no solid
column at their centre, and their volume is carried as canopy drag instead.

**Which buildings are resolved.** A building counts as resolved when at
least half its footprint lies in cells that are solid at 0-10 m. The table
covers the 25 m nest's interior, where the street grid is 18 m cross
streets, 30 m avenues and 60 m deep blocks:

| | 50 m | 25 m |
|---|---|---|
| buildings resolved | 59% (10,481 of 17,903) | 77% (13,805) |
| footprint area in solid cells | 64% | 76% |
| street area in fully open cells | 33% | 46% |
| street area in solid cells | 21% | 14% |
| resolved, under 250 m2 (11,171 buildings) | 54% | 72% |
| resolved, 250-1,000 m2 (5,205) | 65% | 84% |
| resolved, 1,000-2,500 m2 (1,180) | 70% | 93% |
| resolved, 2,500-10,000 m2 (326) | 86% | 97% |
| resolved, over 10,000 m2 (21) | 100% | 100% |

Over the whole 50 m domain, with its six-cell boundary ring left out:

- 10,524 ground columns are solid and 44,412 ground cells are porous;
- the tallest towers stay solid up to 390-400 m (8 columns);
- about a fifth of the buildings are resolved (28,682 of 150,780), which
  is a third of the footprint area. The domain includes the low-rise
  outer boroughs.

So 50 m resolves Manhattan's blocks and large buildings but not its
streets: a cross street narrower than a cell is closed or porous. 25 m
opens most avenues and many cross streets. That is the spacing to use when
the street canyons themselves matter.

**Wind.** Mean speeds over the built cells of the 25 m nest's area, taken
from the 50 m domain:

| height AGL | no buildings: mean (p10-p90) | buildings: mean (p10-p90) | w std, no buildings / buildings | max updraft, no buildings / buildings |
|---|---|---|---|---|
| 10 m | 4.2 (3.4-5.2) m/s | 0.5 (0.05-1.2) m/s | 0.05 / 0.06 m/s | 0.4 / 0.4 m/s |
| 30 m | 5.6 (4.8-6.4) m/s | 1.3 (0.2-2.7) m/s | 0.09 / 0.13 m/s | 0.4 / 0.9 m/s |
| 60 m | 7.0 (6.1-7.7) m/s | 2.9 (0.7-5.2) m/s | 0.17 / 0.23 m/s | 0.6 / 1.8 m/s |

With buildings, the wind follows the street grid:

- it channels along the open avenues and the waterfronts;
- it leaves slack wakes behind single towers that reach up to their heights;
- reversed flow appears in 2.5% of the open cells of the whole 50 m domain
  at 10 m (1.4% at 30 m, 0.4% at 60 m), and in none without buildings;
- up- and down-drafts sit on the tower edges: at 60 m they reach 1.8 m/s up
  and 1.9 m/s down, against 0.6 and 0.7 m/s without buildings.

On the cross-section along 42nd St, the solid towers stand at their
footprint heights, two of them about 185 and 215 m tall mid-section. Above
them the 25 m nest shows rotors and downwash that the 50 m domain smears
out.

In the 19:10 output, the lowest-level flow inside solid cells is 0.045 m/s
at 50 m and 0.023 m/s at 25 m. In open cells it is 2.5 and 1.7 m/s.

**Stations.** Model winds are given first and the station report second.
The Battery heliport (KJRB) at 19:56 UTC reported 4.1 m/s from 200°:

| run | 10 m wind |
|---|---|
| no buildings, 50 m | 4.4 m/s from 205° |
| buildings, 50 m | 3.5 m/s from 235° |
| buildings, 25 m | 3.4 m/s from 254° |

The model wind is taken from the nearest five cells with no building at
0-10 m, averaged over ±10 min. The station sits on the waterfront. The
buildings just inland veer the modelled wind from SSW to WSW, away from
the report.

2 m temperature is 3.0 K below the station's in both runs. The obstacles
exchange no heat (see Limitations). HRRR's 19 UTC start is already 4.8 K
below the station's 18:56 report.

The buildings run stopped at one model hour, so LaGuardia and Central Park
(20:51-20:53) are compared only without buildings:

- LaGuardia: 3.9 m/s from 200° against 4.6 m/s from 160°.
- Central Park: 4.6 m/s from 206° against 8.2 m/s from 240°. Its 2 m
  temperature runs on average 4.1 K below KJRB's in the archive at any time
  shift, so it is used for wind only.

**Cost.** Wall seconds per simulated hour for the whole 4- or 5-domain tree,
on a shared A6000:

| tree | wall s per simulated hour |
|---|---|
| 3 km to 50 m, no buildings | about 6,000 |
| with buildings and the 25 m nest, first 20 minutes | 11,500 (the GPU less busy) |
| with buildings and the 25 m nest, whole hour | 16,200 |

The 50 m domain takes most of the step. GPU memory was 17 GB without the
nest and 21 GB with it.

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
- Resolved obstacles (`obstacle_opt = 1`) act on momentum only. Scalars are
  not masked: advection through a blocked face stops with its wind, but
  sub-grid diffusion across a wall does not. Buildings exchange no heat or
  moisture.
- Buildings in the specified or nested boundary ring of a domain, or
  within two cells of an open edge, are dropped, so the boundary update
  and the obstacles never fight. A periodic axis keeps every cell.
- A moving frame (`umove`, `vmove`) is refused: the buildings are fixed
  to the ground.

- Building heats, shading and moisture are not represented. The product
  describes geometry only.
- With pixels assigned by centre, a cell smaller than about three source
  pixels gets the source's blockiness. GEM-15 at 15 m cannot resolve a 10 m
  street.
- Rotation from the source's UTM grid onto the model axes is taken once,
  at the raster centre.
