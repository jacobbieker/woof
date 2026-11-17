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
tile seams are counted once. Rasters on different lattices must lie at
least one model cell apart: `aggregate` refuses an overlap (that ground
would count twice) and a shared seam (the walls along it would be lost). It then supersamples each raster until each model cell holds at
least nine samples, and assigns each pixel to a model cell by its centre. Walls (height steps between
neighbouring pixels) go to a cell by their midpoint. A WRF domain is read
through its own projection (`ProjectionGrid`); idealised cases use the model
plane (`CartesianGrid`, `block_raster`). Holes (NaN) count as neither
building nor open ground.

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
  `(kappa / ln(d / z0))^2 |U| / spacing`, with d half the spacing and z0
  `obstacle_wall_z0`. This is the wall model WRF-IBM uses (Lundquist et al.
  2010, 2012; Bao, Chow and Lundquist 2018).

All three act as one rate `k = a/dt + b|U|` per face. Each RK stage
subtracts `k / (1 + k dt)` times the coupled momentum from the face's
tendency, after the stage's other slow tendencies. That bounded implicit
form is stable for any k. The acoustic steps carry it as part of the held
slow forcing, so no acoustic kernel changes. The geometry is static (from
`phb`), so a restart rebuilds the same coefficients. One fused kernel
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
| | floor wind / U(2H), roof wind / U(2H) | -0.10, +0.20 | reverse floor flow, forward roof shear |
| | wind inside the buildings / U(2H) | 1.8e-4 | 0 |
| cube, 8 cells per H, periodic 12H x 6H | reattachment behind the cube / H | 2.4 | about 1.6 (Martinuzzi and Tropea 1993), 1.7 (Shah and Ferziger 1997) |
| | horseshoe separation ahead / H | 0.25 | about 1.0 |
| | wind inside the cube / U(2H) | 0.011 | 0 |

The cube's floor is free-slip in this benchmark (no surface layer), so no
boundary layer approaches the cube. That weakens the horseshoe vortex and
lengthens the wake. A real case runs a surface layer on the ground.

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
- Buildings in the specified or nested boundary ring of a domain are
  dropped, so the boundary update and the obstacles never fight.

- Building heats, shading and moisture are not represented. The product
  describes geometry only.
- With pixels assigned by centre, a cell smaller than about three source
  pixels gets the source's blockiness. GEM-15 at 15 m cannot resolve a 10 m
  street.
- Rotation from the source's UTM grid onto the model axes is taken once,
  at the raster centre.
