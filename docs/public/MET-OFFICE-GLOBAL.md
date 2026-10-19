# Met Office Global 10 km forcing

`met-office-global` is the Met Office Unified Model's global deterministic
forecast on its global 10 km standard grid, as published on the AWS open
data registry (registry.opendata.aws/met-office-global-deterministic).
Aliases are `metoffice-global`, `ukmo-global` and `um-global`. It is not
the UK 2 km model, not an ensemble member, and not a model-level product.

Data source: the Met Office. The data are British Crown copyright, the Met
Office, licensed under CC BY-SA 4.0, and supplied on a free, unsupported
basis, not for critical business purposes. **Share-alike applies to derived
outputs**: prepared WRF inputs, forecasts and renders made from these bytes
are CC BY-SA 4.0 too, and must carry that attribution. The land state comes
from NOAA NCEP's GDAS analysis (see below), a United States government
product with no copyright restriction.

## Run one

```sh
woof domain --point 54.5,-3.0 --card "RTX 5070 Ti" --root-dx 9 \
  --hours 6 --source met-office-global --cycle latest --out uk-mo.toml
woof go uk-mo.toml
```

The two halves also run on their own:

```sh
woof fetch --source met-office-global --cycle 2026-10-11T00 --hours 6 --out data/mo
woof prep --source met-office-global --input-list data/mo/inputs.txt ...
```

Use the fetch's `prep-command.txt` rather than listing objects by hand. It
binds two supplements beside the input list:

- `met_office_global_invariant_surface`: the run's step-0
  `height_of_orography` object;
- `physical_analysis_surface_data`: the same cycle's GDAS f000 analysis,
  which the fetch writes to `<out>/donor-gdas/`.

## Only the run's area is decoded

The objects are whole-globe and the bucket has no subsetting service, so
the fetch moves them whole. The mapped engine decodes a NetCDF series
whole as well, so before the input manifest is authored `woof prep` cuts
every state object and the orography to a box around the run (the profile
declares `input_crop = "regular-latlon-netcdf-v1"`, implemented in
`woof.regional_crop`). The box is the experiment's `[fetch] area`, which
`woof domain` and `woof track plan` now write for this source as they do
for GFS; with no area it is the namelist's domains plus 2 degrees. Two
more source cells are kept on every side for the interpolation stencils.
A declared area that does not hold every domain is refused by name.

The crop copies every variable, attribute, dtype and compression setting
and only shortens the latitude and longitude dimensions. A box over the
antimeridian keeps its columns in eastward order and carries on past
180E (181E, 182E, ...), so the cropped axis stays uniform. The cropped
objects, a `crop-receipt.json` (input and output SHA-256 of every object,
the box and the index plan) and the input manifest that binds them go in
a `met-office-global-crop-<key>/` folder keyed by the request, beside
where the manifest was asked for; an intact crop of the same request is
reused. In a standalone `woof prep` that folder sits in the fetch folder,
one per domain prepared from it. A 27-hour, 3-hourly window over a domain
about 90 km wide went from 9.1 GB of objects to 7.9 MB, and the engine's
resident memory from an out-of-memory kill at 68.7 GB to under 1 GB (the
whole preparation peaked under 4.5 GB). The source is not prepared as
posted, so every lead is cropped before the decode.

The engine's NetCDF decode is still not memory-priced: it decodes the
whole series before carving it by valid time, logs `memory_priced: false`
with `per_time_bytes: 0`, and refuses under a declared
`GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES`. Pricing it needs a per-time
NetCDF decode in the engine (the GRIB2 path's sliced stream); after the
crop, the whole-series decode is a few megabytes.

The GDAS donor is not cropped: it is one GRIB2 analysis whose land and
soil records are decoded once, and the composition requires the donor to
cover the whole primary grid, which a global analysis always does. The `area` does
not key the managed download folder for a whole-object route, so moving
or re-planning a domain reuses the objects already fetched.

## What the source gives you

The bucket is `met-office-atmospheric-model-data` (eu-west-2, anonymous
HTTPS). Every object is one NetCDF-4 variable at one lead:

```
global-deterministic-10km/<RUN YYYYMMDDTHHMMZ>/<VALID YYYYMMDDTHHMMZ>-PT<HHHH>H00M-<variable>.nc
```

The grid is a regular latitude/longitude grid of 2560 x 1920 cell centres,
0.140625 degrees in longitude by 0.09375 in latitude (about 10 km at
mid-latitudes). Time, the forecast reference time and the forecast period
are 0-d scalar variables in each object. The profile reads the cycle from
`forecast_reference_time`, so every lead of a run is one cycle.

| WRF input | Met Office object | notes |
|---|---|---|
| temperature, geopotential height | `temperature_on_pressure_levels`, `geopotential_height_on_pressure_levels` | 37 levels, 1000 to 0.4 hPa |
| u, v | `wind_speed_on_pressure_levels`, `wind_direction_on_pressure_levels` | from speed and meteorological from-direction |
| specific humidity | `relative_humidity_on_pressure_levels` | 33 levels, 1000 to 10 hPa, through `specific_humidity_from_rh` |
| surface pressure | `pressure_at_mean_sea_level` + `height_of_orography` | rebuilt at every lead (below) |
| terrain | `height_of_orography` | published at step 0 only, carried to every lead |
| skin temperature | `temperature_at_surface` | |
| 2 m temperature and humidity | `temperature_at_screen_level`, `relative_humidity_at_screen_level` | the screen is 1.5 m and is used as published (`screen_height_m: 1.5`) |
| 10 m wind | `wind_speed_at_10m`, `wind_direction_at_10m` | |
| snow water equivalent | `snow_depth_water_equivalent` | metres of water, times 1000 kg m-3 |
| soil temperature and moisture, land mask, sea ice | not published | borrowed from the same-cycle GDAS analysis (below) |

The pressure ladder stops at **10 hPa**. Humidity is published only to
10 hPa, while temperature, height and wind continue to 0.4 hPa; the profile
declares the 33 levels where all four exist and cuts the four upper levels
from the 37-level objects by value. That is above the 50 hPa default model
top, so the wizard's default stands.

The pressure-level and near-surface objects use the same variable names
(`air_temperature`, `relative_humidity`, `wind_speed`,
`wind_from_direction`). Each selector pins the CF `coordinates` attribute,
which lists the scalar `height` coordinate only on the near-surface
objects, so a 1.5 m field can never be read as a 1000 hPa one.

Forcing is a uniform three-hour series: `interval_seconds` is 10800 in the
companion `namelist.wps`. The 00 and 12 UTC runs post hourly to f054,
3-hourly to f144 and 6-hourly to f168; the 06 and 18 UTC runs post hourly
to f054 and 3-hourly to f069. The three-hour series this route prepares
therefore ends at f144 (00/12Z) or f069 (06/18Z).

## Snow over the ice sheets is capped at init

Over the Greenland ice sheet the product's `snow_depth_water_equivalent` is
the snow store of its ice-sheet tiles, about 51.8 m of water: 51,800 kg m-2
on the 2026-10-11 00Z run, up to 68,800 kg m-2 further north.  At the 5:1
depth rule both WRF real and MPAS init use, that is 259 m of snow.  The
value is passed on as published.  The initial state then caps snow water
equivalent at 5000 kg m-2, the ceiling WRF's Noah-MP initialisation applies
(`NOAHMP_INIT`), and scales the snow depth by the same factor so the density
is kept.

The cap is never silent:

- **WRF route.** `woof prep` logs `snow cap: capped SNOW at 5000 kg m-2 on K
  of N cell(s) (source value up to X kg m-2, ...)`. The mapped preparation proof
  carries an `init_caps` block with the cap, the count, the largest value
  and its bounding box. The init's unit-error ceiling of 20,000 kg m-2
  (`_SNOW_WATER_CEILING_KG_M2` in `woof/ingest/soil.py`) used to refuse this
  source over Greenland outright. It still applies everywhere else. On land
  ice (soil category 16), and on any patch of such values that touches land
  ice (the ice margin as interpolation smears it), an ice-sheet store is
  admitted up to 200,000 kg m-2 and then capped.
- **hex.** `woof hex intermediate` logs and records (`init_caps_preview`)
  how many source points exceed 5000 kg m-2. `woof hex init` prints
  `rw_mpas_init: capped SNOW at 5000 kg m-2 on K cell(s) ...` and stores the
  counts in its receipt (`init_caps`). On the Greenland 100 m cull, all
  24,133 cells were capped, from up to 51,949 kg m-2.

The cap applies to every source. GFS and GDAS never reach it; ECMWF IFS,
ERA5 and ICON do on permanent snow. The soil moisture borrowed from GDAS
is capped at the cell's porosity in the same way (see
[ICON-GLOBAL.md](ICON-GLOBAL.md)).

## Surface pressure

The product publishes no surface pressure. At every lead it is rebuilt from
that lead's mean-sea-level pressure at the step-0 orography, through the
same lead's isobaric heights, by WRF real's `sfcprs3` relation
(`surface_pressure_from_sea_level`). The column mass follows the forecast;
nothing about surface pressure is borrowed. On the 2026-10-11 00Z run over
the UK it equals MSLP to the pascal over the sea and falls by about
11.8 Pa per metre of terrain.

## The land state comes from GDAS

The Met Office product has no soil, no land-sea mask and no sea ice. The
profile borrows them from the **same cycle's** NCEP GDAS 0.25-degree
analysis through the cross-source composition, the same mechanism `aigfs`
and `aigefs` use:

- land fraction (GDAS `LAND`) and sea ice (GDAS `ICEC`);
- the four Noah soil layers, 0-10, 10-40, 40-100 and 100-200 cm, for both
  temperature and volumetric moisture.

They are decoded through a packaged GDAS donor table
(`rw-wps-gdas-pgrb2-land-ice-donor.mapping.json`, the shipped GDAS donor
plus one sea-ice row), pinned by SHA-256 in the composition. The clock is
`source_cycle_analysis_broadcast`. The donor must be the one analysis
record whose valid time is the Met Office cycle, and it is carried
unchanged to every lead. A donor from another cycle refuses by name.

The two grids share no cells, so the binding uses the
`nearest_coordinate_index` grid alignment. Each Met Office row and column
takes the nearest GDAS row and column, axis by axis, modulo 360 in
longitude, with ties going to the lower index. Nothing is interpolated:
every borrowed value is a GDAS cell value, repeated about 2 x 3 times across
the finer grid. **The borrowed coastline is GDAS's 0.25-degree one, not the
Met Office's.** The composition receipt states this in its
`resolution_note`. The GDAS file must cover the whole Met Office grid, so a
regional GDAS subset refuses.

Soil, mask and ice are frozen at the analysis while the Met Office
atmosphere advances. For a regional run of a few days this is the same
class of approximation as WRF's own default of not updating SST.

## Older runs

`height_of_orography` exists from the 2025-12-11 12Z run onward. Earlier
runs have no step-0 orography, so there is no terrain to bind and they
refuse by naming the missing terrain supplement. Terrain is never taken
from elsewhere.

From 2025-12-11 12Z through 2026-01-21 06Z the 3-D height object is named
`height_ASL_on_pressure_levels`; `geopotential_height_on_pressure_levels`
replaces it from 2026-01-21 12Z. The older object's variable carries the
CF standard name `geopotential_height`, so the height selector resolves it
by standard name, and the decode reports that it did. No run from that
middle period has been prepared end to end.

## Readiness

The registry records this source as runnable, not certified. The
2026-10-11 00Z run, f000 and f003, prepared end to end on a 100 x 80 UK
domain at 9 km through the Rust engine (decode and compose 215 s, the whole
preparation 286 s). Over flat coast the prepared surface pressure sits
1.3 Pa below the Met Office MSLP on average. The 10 m winds correlate at
0.996 (u) and 0.988 (v) with the published speed and direction. WOOF's
separate stock-WRF acceptance status is not claimed for it.

What this route does not do: model levels (none are published), vertical
velocity (published but not bound; the explicit zero applies), physical
snow depth (not published), or the UK 2 km model.
