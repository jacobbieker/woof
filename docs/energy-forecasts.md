# Energy forecasts along power lines, substations and renewables

`woof energy` builds high-resolution forecasts (typically 100 m or 50 m grid
spacing) along grid assets: overhead lines, cables, substations, wind farms
and solar farms. It reads grid topology from OpenStreetMap or from your own
files, turns the assets into forecast sites, plans the model domains that
cover them, runs those domains, and samples the output back at the sites.

The point is to forecast the weather an asset actually sees, rather than the
weather at the nearest point of a 3 km grid:

- **Dynamic line rating (DLR).** Conductor cooling depends on the wind
  component perpendicular to the line, the air temperature and the sunshine
  at the span. A ridge or a valley a few hundred metres across changes all
  three.
- **Conductor icing.** Supercooled cloud water and freezing rain accrete on
  conductors and towers. The cloud base and the 0 °C level are often
  decided by terrain at sub-kilometre scales.
- **Wind power** at turbine hub heights.
- **PV power** from global, direct and diffuse irradiance.

Every stage reads and writes one versioned file, so you can rerun, replace or
feed any stage from outside without touching the others.

```text
woof energy fetch / import  ->  assets.geojson   woof-energy.assets.v1
woof energy sites           ->  sites.json       woof-energy.sites.v1
woof energy plan            ->  plan/plan.json   woof-energy.plan.v1  (+ emitted configs)
woof energy run             ->  one run per plan domain (runs/manifest.json)
woof energy extract         ->  forecast.nc      woof-energy.forecast.v1
woof energy rating          ->  products.nc
```

`woof energy` with no subcommand prints this overview. Every flag is listed
under the `woof energy` headings of
[the CLI options reference](public/CLI-OPTIONS.md).

## Data sources and licences

| Source | How to read it | Licence | Obligation |
|---|---|---|---|
| OpenStreetMap power data, via the Overpass API | `woof energy fetch` | ODbL 1.0 | Attribution "(c) OpenStreetMap contributors"; share-alike applies to a redistributed derived database such as `assets.geojson` |
| PyPSA-Eur network CSVs (buses, lines, links, transformers, generators) | `woof energy import --format pypsa-eur` | CC-BY 4.0 | Attribution |
| UK Renewable Energy Planning Database (REPD) CSV | `woof energy import --format repd` | Open Government Licence v3 | Attribution |
| Your own assets as GeoJSON or CSV | `woof energy import --format geojson` or `--format csv` | Yours | Yours |

[OpenInfraMap](https://openinframap.org) is a map rendering of the same
OpenStreetMap `power=*` data. `woof energy fetch` does not read OpenInfraMap
itself. It asks an Overpass server for the OSM tags OpenInfraMap draws:
`power=line`, `minor_line`, `cable`, `substation`, `plant`, `generator` and
`tower`.

Each assets document records its provenance in the `sources` list of its
top-level `woof` member: the source, the licence, the retrieval time, the
endpoint or URL, the query and the attribution text. Those records travel
with the file when it is merged or shared. If you publish maps or tables
made from OSM data, show the attribution.

Overpass responses are cached under `~/.woof/cache/energy-osm/`. Set
`WOOF_ENERGY_OSM_CACHE` to move the cache. `--offline` uses only the cache
and refuses if any tile is missing. `--refresh` revalidates cached
responses. `--endpoint` names the Overpass interpreter to ask first, before
the built-in mirror ladder. Public Overpass servers are shared and
rate-limited, so fetch a region once and work from the cache.

## The file contracts

Each document carries a `schema` string and is refused on read when the
string, a required field or a value range is wrong. Nothing is silently
coerced. The contracts are defined in `woof/energy/contracts.py`.

### `woof-energy.assets.v1` (`assets.geojson`)

A GeoJSON FeatureCollection in EPSG:4326 (longitude, latitude), with one
Feature per asset. Feature properties:

| Property | Meaning |
|---|---|
| `asset_id` | Stable, source-qualified identifier: `osm:way/123`, `pypsa-eur:line/8`, `repd:4567` |
| `kind` | `line`, `minor_line`, `cable`, `substation`, `plant`, `generator` or `tower` (named after the OSM `power=*` values) |
| `source`, `source_ref` | Where the asset came from, and its identifier there |
| `name`, `operator` | As tagged |
| `voltage_kv` | Every voltage a line or substation carries, highest first |
| `circuits`, `cables`, `frequency_hz` | As tagged |
| `generator_source` | `wind`, `solar`, `hydro`, `tidal`, `wave`, `gas`, `oil`, `coal`, `nuclear`, `biomass`, `biogas`, `waste`, `geothermal`, `battery` or `other` |
| `capacity_mw`, `hub_height_m`, `rotor_diameter_m` | Generator attributes, when known |
| `license` | The asset's licence |
| `tags` | Any further source attributes, as strings |

Lines, minor lines and cables are LineStrings or MultiLineStrings. Towers are
Points. Substations, plants and generators are Points, Polygons or
MultiPolygons.

### `woof-energy.sites.v1` (`sites.json`)

The points where a forecast is wanted. The file is columnar (one list per
field), because a national grid sampled every 100 m has a few hundred
thousand points.

| Column | Meaning |
|---|---|
| `site_id`, `asset_id` | The site, and the asset it samples |
| `kind` | `line_sample`, `tower`, `substation`, `turbine`, `pv` or `plant` |
| `lat`, `lon` | Position, degrees |
| `bearing_deg` | Conductor azimuth at the site, clockwise from true north, in [0, 360); null where there is no conductor |
| `chainage_m` | Distance along the asset |
| `voltage_kv`, `hub_height_m`, `capacity_mw` | Carried from the asset, when known |

`heights_m` applies to the whole set: the heights above ground at which every
site is sampled. Turbine hub heights are added to it.

### `woof-energy.plan.v1` (`plan/plan.json`)

The domains a planner emitted. Paths are relative to the plan file's
directory. Each domain has:

- `domain_id`, `topology`, `role` (`parent`, `child` or `mesh`) and `dx_m`
- `config` and `wps_namelist`: the emitted WOOF experiment TOML and its
  `namelist.wps` (WRF topologies), with the WRF `grid_id` inside that config
- `parent`: the domain whose output forces this one (`wrf-tiles` children only)
- `run_dir` and `output_glob`: where `woof energy run` writes history, and
  the glob (`wrfout_d02_*`, or `history.*.nc` for MPAS) the extractor reads
- `footprint`: the domain's mass-point extent as a closed lon/lat ring
- `site_ids`: the sites this domain owns. Each site has at most one owner,
  the finest domain that covers it.
- `mesh`: the MPAS mesh documents (`hex-swath` only)
- `extra`: planner-specific data the run orchestrator consumes, such as
  `downscale_args` for a `wrf-tiles` child and `commands` for a `hex-swath`
  mesh

The plan also records the start time, the length, the forcing source, a
binding (path and SHA-256) of the sites document, and notes on any choice the
planner made for you.

### `woof-energy.forecast.v1` (`forecast.nc`)

A netCDF file (or a Zarr, Icechunk or CSV export) with dimensions
`(time, site, height)`. Winds are earth-relative. `wind_from_direction` is
meteorological: the direction the wind blows from, clockwise from true north.
`line_normal_wind` is the absolute wind component perpendicular to the site's
`bearing_deg`, and NaN where the site has no bearing. `wind_attack_angle` is
the acute angle between the wind and the conductor axis, from 0 to 90
degrees. Those two are the inputs conductor-cooling models need.

Data variables (`FORECAST_VARIABLES`):

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `u` | time, site, height | m s-1 | eastward wind |
| `v` | time, site, height | m s-1 | northward wind |
| `w` | time, site, height | m s-1 | upward air velocity |
| `wind_speed` | time, site, height | m s-1 | wind speed |
| `wind_from_direction` | time, site, height | degree | wind from direction |
| `line_normal_wind` | time, site, height | m s-1 | wind component normal to the conductor |
| `wind_attack_angle` | time, site, height | degree | angle between wind and conductor axis |
| `air_temperature` | time, site, height | K | air temperature |
| `air_pressure` | time, site, height | Pa | air pressure |
| `air_density` | time, site, height | kg m-3 | air density |
| `specific_humidity` | time, site, height | kg kg-1 | specific humidity |
| `relative_humidity` | time, site, height | % | relative humidity |
| `cloud_liquid_mixing_ratio` | time, site, height | kg kg-1 | cloud liquid water mixing ratio |
| `rain_mixing_ratio` | time, site, height | kg kg-1 | rain mixing ratio |
| `ice_mixing_ratio` | time, site, height | kg kg-1 | cloud ice mixing ratio |
| `snow_mixing_ratio` | time, site, height | kg kg-1 | snow mixing ratio |
| `t2` | time, site | K | 2 m air temperature |
| `q2` | time, site | kg kg-1 | 2 m water vapour mixing ratio |
| `rh2` | time, site | % | 2 m relative humidity |
| `u10` | time, site | m s-1 | 10 m eastward wind |
| `v10` | time, site | m s-1 | 10 m northward wind |
| `wind_speed_10m` | time, site | m s-1 | 10 m wind speed |
| `psfc` | time, site | Pa | surface pressure |
| `ghi` | time, site | W m-2 | global horizontal irradiance |
| `dni` | time, site | W m-2 | direct normal irradiance |
| `dhi` | time, site | W m-2 | diffuse horizontal irradiance |
| `cos_solar_zenith` | time, site | 1 | cosine of solar zenith angle |
| `precipitation_rate` | time, site | kg m-2 s-1 | total precipitation rate |

Coordinates (`FORECAST_COORDINATES`):

| Coordinate | Dimensions | Units | Meaning |
|---|---|---|---|
| `time` | time | - | valid time (UTC) |
| `height` | height | m | height above ground level |
| `site_id` | site | - | site identifier |
| `asset_id` | site | - | asset identifier |
| `kind` | site | - | site kind |
| `lat` | site | degree_north | latitude |
| `lon` | site | degree_east | longitude |
| `bearing_deg` | site | degree | conductor azimuth |
| `terrain_height` | site | m | model terrain height |
| `domain_id` | site | - | owning plan domain |
| `dx_m` | site | m | grid spacing of the owning domain |
| `inside` | site | 1 | 1 where the owning domain sampled the site |
| `voltage_kv` | site | kV | asset voltage (NaN where unknown) |
| `hub_height_m` | site | m | turbine hub height above ground (NaN where unknown) |
| `capacity_mw` | site | MW | asset generating capacity (NaN where unknown) |

Each file also carries the global attributes `schema`, `plan_sha256` and
`sites_sha256`. Those attributes bind the forecast to the exact plan and
sites it came from. A site its owning domain could not sample has
`inside = 0` and NaN data. Nothing is extrapolated.

## The pipeline, stage by stage

### 1. Assets: `fetch` and `import`

`woof energy fetch` takes an area as `--bbox W,S,E,N` or `--polygon
FILE.geojson`. A negative west longitude must be written with an equals sign,
`--bbox=-4.2,51.5,-3.3,51.8`. Otherwise the shell parser reads `-4.2` as a
flag. `--kinds` selects asset kinds (default
`line,cable,substation,plant,generator`; add `tower` for tower nodes).
`--min-voltage-kv` drops lines, cables and substations below a voltage.
Assets with no voltage tag are kept rather than guessed. `--timeout-s` is the
server-side Overpass timeout per tile.

`woof energy import` reads one or more files in one `--format`. With
`--merge ASSETS.geojson` it adds the new assets to an existing document and
drops duplicates by source reference and proximity. For a generic CSV, name
the columns with `--lat-col`, `--lon-col` and `--id-col`, and give `--kind`
when the file does not say what each row is.

### 2. Sites: `sites`

`woof energy sites` samples lines and cables every `--spacing-m` metres
(default 100) and records the local conductor bearing at each sample. It adds
a site at each substation, turbine (with its hub height), PV farm and plant.
`--include-towers` adds a site at every tower node. `--pv-grid-m` samples a
solar-farm polygon on a grid instead of at its centroid. `--kinds`,
`--min-voltage-kv` and `--region FILE.geojson` filter the sites.
`--heights-m` sets the heights above ground (default `10,30,100`).

Match `--spacing-m` to the grid. Sites much closer together than `--dx-m`
sample the same grid cells and add file size without adding information.

### 3. Domains: `plan`

`woof energy plan SITES.json --topology T -o DIR` writes `DIR/plan.json` and
every configuration it emits. The common flags:

| Flag | Meaning |
|---|---|
| `--dx-m` | Target grid spacing over the sites (default 100) |
| `--corridor-km` | Half-width of the high-resolution corridor around each site (default 2) |
| `--parent-dx-m` | Outer parent grid spacing (default: chosen by the planner) |
| `--start`, `--hours` | Forecast start (`YYYY-MM-DDTHH`, UTC; default the most recent 00/06/12/18 cycle) and length (default 24) |
| `--source` | Initial and boundary condition source (default: the one `woof domain` emits) |
| `--card` or `--vram-gib` | Size each domain for a GPU tier or a memory budget |
| `--max-domains` | Refuse a plan with more high-resolution domains than this |
| `--nz` | Vertical levels (default: the planner's ladder) |

Emitted WRF configurations write history with the `energy` output preset
(`[output] preset = "energy"`), which keeps the fields the extractor reads,
including the direct normal (`SWDDNI`) and diffuse (`SWDDIF`) irradiance
behind `dni` and `dhi`.

### 4. Running: `run`

`woof energy run PLAN.json` runs every domain in parent-first order:
`woof go` for a WRF root, `woof downscale` for a `wrf-tiles` child after its
parent has finished, and the `woof hex` route for a `hex-swath` mesh. It
records a run manifest at `runs/manifest.json` next to the plan.
`--dry-run` prints the commands without running them. `--only ID[,ID]` runs
named domains whose parents have already run. `--resume` skips domains the
manifest records as complete.

### 5. Sampling: `extract`

`woof energy extract PLAN.json -o forecast.nc` reads each domain's history,
samples it at the sites that domain owns, and writes one `forecast.v1` file.
Horizontal interpolation is bilinear on mass points. Vertical interpolation
is linear in height above model terrain, between mass levels. `--sites`
overrides the sites document the plan names, `--heights-m` overrides its
heights, and `--vars` selects variables. `--format` is `netcdf` (the
default), `zarr`, `icechunk` or `csv`. Zarr needs `xarray` and `zarr`, and
Icechunk needs `icechunk` as well. Neither is a core dependency.

The heavy lifting is the Rust site sampler, `librw_sitesample.so`, built
from the `tools/rustwx` workspace. `WOOF_SITESAMPLE_BRIDGE` points WOOF at a
specific build. `woof doctor` reports whether it is found.

### 6. Products: `rating`

`woof energy rating FORECAST.nc -o products.nc` computes the products named
in `--products` (default `dlr,icing,wind-power,pv-power`). See
[Products and their assumptions](#products-and-their-assumptions).

## Choosing a topology

| | `wrf-nests` | `wrf-tiles` | `hex-swath` |
|---|---|---|---|
| Shape | One WRF run with sibling nests over corridor clusters | One regional parent run plus offline child tiles | One MPAS variable-resolution mesh, refined along the corridors and culled to a limited area |
| Domain count | At most 20 child domains (WRF's `max_dom` of 21 including the root) | No limit | One mesh |
| Coupling | Two-way inside one run | One-way; each tile is forced by archived parent history | One mesh, no nest boundaries |
| Best for | A compact cluster of lines or farms | Long corridors and national grids | Experiments with a single seamless corridor mesh |

### `wrf-nests`

Every nest runs inside one forecast and is forced by its parent every step.
Use it when the sites fall into a handful of clusters, for example one
substation and the lines leaving it, or a wind farm and its export cable.
The planner refuses a plan that needs more than 20 child domains. If it
inserts intermediate nests between the parent and the 100 m leaves, those
count toward the 20 as well.

### `wrf-tiles`

The irregular WRF topology. One regional parent runs first. The planner then
covers the corridors with any number of rectangular child tiles, each run
afterwards by `woof downscale` from the parent's archived history. The
tiles do not have to line up with each other or tile the whole domain.
Together they follow the line. Because the tiles are separate runs there is
no domain-count ceiling, and tiles can run on separate GPUs or machines in
any order once the parent is done.

The costs are those of offline downscaling, listed in the
[CLI user manual](public/CLI-USER-MANUAL.md) and [DOWNSCALE.md](public/DOWNSCALE.md).
The child sees the parent only at its history interval, and nothing feeds
back to the parent. Each tile also has its own lateral boundary zone, which
is one reason to keep `--corridor-km` wide enough that the sites sit well
inside their tile.

### `hex-swath`

An MPAS mesh whose cell spacing falls to `--dx-m` along the corridors and
rises to `--parent-dx-m` away from them, culled to the region. It is built
with the mesh generator behind `woof hex`, and it must clear the same
generation gates as any other mesh (`woof/hex/mesh_spec_gates.py`):

- **Short dual edge floor.** The floor is 400 coordinate quanta of the
  storage the mesh is written in. That is 200 m for a mesh with a native
  MPAS-A counterpart (binary32 coordinates). A mesh that `rw_mpas_mesh`
  generates is stored at binary64, where 400 quanta is about 3.7e-7 m, so a
  generated 50 m or 100 m corridor mesh is not refused by edge length.
- **Dual-to-primal edge ratio.** The shortest `dvEdge/dcEdge` must be at
  least 0.02.
- **Transition-band gate.** The steepest requested spacing gradient must be
  at most 12.25 % per cell. That is at least six cells across each doubling
  of spacing. This is the gate a fine corridor runs into. From a 100 m
  corridor to a 25.6 km background is eight doublings, and every doubling
  needs its own transition band. A narrow `--corridor-km` with a coarse
  background is refused before anything is built, and the refusal names the
  ramp widening that clears it.

The finest graded mesh this tree has a measurement for is 0.75 km. Meshes at
100 m and 50 m are unmeasured.

A hex corridor can be forced one way from a WOOF WRF run, such as a 1 km to
3 km `wrf-nests` or `wrf-tiles` parent, instead of from a GRIB source. Run
`woof hex intermediate` with these flags:

- `--source wrfout`.
- `--wrfout-glob`, quoted, naming the history of the finest parent domain
  that covers the cull. For a `wrf-nests` parent that is the nest's
  `wrfout_d02_*` (or deeper), not the coarse `wrfout_d01_*`.
- `--cull-region`, naming the plan's `cull_region.json`.
- `--halo-km`, set to the boundary-ring width the plan reports. The region
  file holds only the cut, and the rings lie outside it.
- `--out-dir`, a fresh directory. A directory that already holds
  intermediates is refused.

This writes one WPS intermediate per wrfout time on a regular lat-lon grid at
the parent's dx. Every WRF mass level is kept, with its 3-D pressure, plus the
four Noah soil layers. Winds are rotated to earth-relative on the WRF grid
before they are moved. The cull, plus `--margin-km` of margin, must sit inside
the parent's interior: `--wrf-edge-cells` (default 5) relaxed boundary rows
are excluded on every side, and a cull that reaches them is refused. A parent
that is not on Lambert, polar stereographic or Mercator is refused, as is a
parent whose soil column is not Noah or Noah-MP, or a set of files that
repeats a valid time. The receipt records the sha256 of every wrfout it read.
`woof hex lbc` then builds the boundary series from those files.

## Resolution: 100 m or 50 m

### Turbulence: the gray zone and LES

Between about 1 km and about 100 m, a model grid partly resolves the largest
boundary-layer eddies. This is the "gray zone" or "terra incognita". A 1-D
PBL scheme assumes all of the turbulence is subgrid. An LES closure assumes
most of it is resolved. At this range neither assumption holds.

- At **100 m**, run the high-resolution domain as LES: `km_opt = 2` or `3`
  with `diff_opt = 2` and `bl_pbl_physics = 0`. WOOF's LES closures have been
  measured at 100 m on an idealized convective boundary layer and as a 250 m
  nest in a real terrain-following tree. Read [LES.md](public/LES.md) for
  what has been measured, the vertical-level limits (115 levels at the usual
  `p_top` under RRTMGP) and what is still open.
- At **50 m** the same recipe applies, but WOOF has no measurement at 50 m.
- The parent chain crosses the gray zone. The shipped nested-LES trees run
  a 750 m parent with a 1-D PBL scheme above the LES child.
  [GRAYZONE-NEST.md](public/GRAYZONE-NEST.md) describes the variant whose
  parent runs the scale-aware Shin-Hong scheme, and its open findings.

On stable, windy winter nights, which matter most for icing and often for
DLR, a 100 m grid still does not resolve the smallest eddies. The forecast
wind is a grid-cell average, not a gust at a span.

### Static fields: terrain and land cover

Domains at 1 km or finer take Copernicus DEM GLO-30 terrain by default. In
the United States you can choose USGS 3DEP at about 10 m. See
[HIGHRES-TERRAIN.md](public/HIGHRES-TERRAIN.md).

- At **100 m**, GLO-30 (about 30 m) puts roughly three source pixels across
  each grid cell. That is enough.
- At **50 m**, GLO-30 is marginal: fewer than two source pixels across a
  cell, so the grid adds little terrain detail beyond 100 m. Where 3DEP
  covers the area, use it.
- The default high-resolution land cover (CGLC-MODIS-LCZ) is 100 m, and
  SoilGrids soil texture is 250 m. At 50 m, land use and soil are coarser
  than the grid.

### Time step and cost

WOOF's real-data starting clock is 5 s per kilometre of grid spacing: 0.5 s
at 100 m and 0.25 s at 50 m. Steep terrain can shorten it further (see
`terrain_clock` in [CONFIGURATION.md](public/CONFIGURATION.md)).

Cost over a fixed area grows with the cube of the refinement. Halving the
spacing gives four times the columns and twice the steps, so a 50 m corridor
costs about eight times as much as a 100 m one of the same footprint. A
corridor of half-width `--corridor-km` around a 100 km line covers 400 km²
at the default of 2 km. That is 40,000 columns at 100 m and 160,000 at 50 m.
Reduce `--corridor-km` before you reduce `--dx-m`, and size the plan with
`--card` or `--vram-gib` for the GPU you will actually run on.

## Worked example: a 400 kV corridor in South Wales

This example plans a 100 m forecast along the 400 kV lines across South Wales
and runs it with each topology. Paths are relative to a working directory of
your choice.

**1. Fetch the assets** (OpenStreetMap, cached after the first call):

```bash
woof energy fetch --bbox=-4.2,51.5,-3.3,51.8 --kinds line,cable,substation,plant,generator --min-voltage-kv 132 -o wales/assets.geojson
```

**Optionally, merge renewable sites** from a downloaded REPD CSV:

```bash
woof energy import repd.csv --format repd --merge wales/assets.geojson -o wales/assets-repd.geojson
```

**2. Make the sites.** This keeps the 400 kV lines, samples them every 100 m,
and asks for 10 m, 30 m and 50 m above ground (conductors on 400 kV lattice
towers typically hang between about 10 m and 45 m above ground):

```bash
woof energy sites wales/assets.geojson --kinds line --min-voltage-kv 400 --spacing-m 100 --heights-m 10,30,50 -o wales/sites.json
```

**3. Plan the domains.** Pick one topology.

One run with sibling nests over the line clusters:

```bash
woof energy plan wales/sites.json --topology wrf-nests --dx-m 100 --corridor-km 2 --start 2026-10-10T00 --hours 24 --vram-gib 24 -o wales/plan-nests
```

One regional parent plus offline tiles along the lines:

```bash
woof energy plan wales/sites.json --topology wrf-tiles --dx-m 100 --corridor-km 2 --parent-dx-m 900 --start 2026-10-10T00 --hours 24 --vram-gib 24 -o wales/plan-tiles
```

An MPAS corridor mesh:

```bash
woof energy plan wales/sites.json --topology hex-swath --dx-m 100 --corridor-km 3 --start 2026-10-10T00 --hours 24 --vram-gib 24 -o wales/plan-hex
```

Read `plan.json` and the emitted configurations before running. The notes
list every choice the planner made: the parent spacing, the ladder, and any
refusal it worked around.

**4. Run it.** Print the commands first:

```bash
woof energy run wales/plan-tiles/plan.json --dry-run
woof energy run wales/plan-tiles/plan.json
```

If a tile fails, rerun only what is left with
`woof energy run wales/plan-tiles/plan.json --resume`.

**5. Sample the sites:**

```bash
woof energy extract wales/plan-tiles/plan.json -o wales/forecast.nc
```

**6. Compute line ratings and icing:**

```bash
woof energy rating wales/forecast.nc --products dlr,icing -o wales/products.nc
```

`--conductor auto` (the default) picks a conductor from the built-in table by
line voltage. Name one with `--conductor NAME`, or supply your own table with
`--conductor-table FILE.json`.

The same steps fit a national grid. Fetch a larger box or a `--polygon`, use
`--topology wrf-tiles`, and set `--max-domains` as a guard on how many tiles
you are prepared to run.

## Products and their assumptions

`woof energy rating` writes one netCDF file with a variable per product on
the forecast's `(time, site)` or `(time, site, height)` grid. It records the
constants it assumed in the file's attributes. These are engineering models
driven by forecast weather. They are not a substitute for an operator's own
rating methodology.

| Product | Model | Forecast inputs | Assumptions to check |
|---|---|---|---|
| `dlr` | IEEE 738 steady-state conductor heat balance: convective and radiative cooling against solar and Joule heating | `line_normal_wind`, `wind_attack_angle`, `air_temperature`, `air_density`, `ghi`, `dni`, `dhi`, `bearing_deg` | The conductor (diameter, resistance, maximum temperature, emissivity and absorptivity) from `--conductor` or the table; sites with no bearing get no rating |
| `icing` | Makkonen accretion on a cylinder, with ISO 12494 conventions | `cloud_liquid_mixing_ratio`, `rain_mixing_ratio`, `air_temperature`, `wind_speed`, `air_density` | An assumed droplet median volume diameter (MVD), because the model's microphysics does not forecast one; a reference conductor diameter |
| `wind-power` | A generic IEC-style power curve, density-corrected | `wind_speed` and `air_density` at hub height, `capacity_mw`, `hub_height_m` | One generic curve for every turbine, scaled by capacity; no wake losses, availability or curtailment |
| `pv-power` | A PVWatts-like plane-of-array and cell-temperature model | `ghi`, `dni`, `dhi`, `cos_solar_zenith`, `t2`, `wind_speed_10m`, `capacity_mw` | Assumed tilt, azimuth, temperature coefficient and system losses; no shading, soiling or inverter clipping detail |

## Limitations

- **Grid data completeness.** OpenStreetMap power data is extensive in much
  of Europe but uneven. Voltages, circuits and operators are often missing,
  and some underground cables are not mapped. Check the assets before relying
  on a corridor.
- **Model terrain.** Heights are above model terrain, which is smoothed. A
  conductor that crosses a valley at 60 m above its floor is sampled at a
  height above the grid-cell terrain, not above the valley floor.
- **Sub-grid sheltering.** Trees, buildings and small terrain features below
  the grid spacing are represented only through roughness. Local wind at a
  span can differ from the grid-cell value.
- **One deterministic forecast.** The pipeline does not produce probabilistic
  ratings. Treat a DLR forecast as one realisation, not a guaranteed
  capacity.
- **Cables.** IEEE 738 is an overhead-conductor model. A buried cable's
  rating depends on soil temperature and thermal resistivity, which this
  pipeline does not model.
- **Offline tiles** (`wrf-tiles`) are one-way and see the parent only at its
  history interval. A short parent history interval costs disk but improves
  the tiles' boundaries.
- **`hex-swath` at 50 m to 100 m** is unmeasured. See
  [Choosing a topology](#choosing-a-topology).
- **50 m in general** is unmeasured in WOOF, for LES and for static fields.

## Checking an install

`woof doctor` has a non-blocking row for `woof energy`. It reports whether
the package imports, whether every stage is implemented in this build, the
size of the Overpass cache, whether the site sampler
library is on disk (and the build command if it is not), and whether the
optional `xarray`, `zarr` and `icechunk` packages import. It contacts no
server. `woof doctor --explain` prints the full evidence.
