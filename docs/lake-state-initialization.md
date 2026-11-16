# Lake initialization sources

`RunConfig.lake_state_source` defaults to `auto`. Before the land-surface cold
start, public NOAA GLSEA temperature and ice concentration replace valid lake
cells within the source raster. Other cells retain their forcing. The table
`woof/data/surface_obs/lake-state-sources.v1.toml` owns identifiers, dates,
variables, units, conversions, URLs, coverage, age limits and licences. Additional
regular CF sources are table rows. Downloads are cropped and cached; no
credentials are discovered. GLSEA is free to use and redistribute under its
NOAA metadata terms.

Use these shared experiment settings for the coupled lake model:

```toml
[shared]
sf_lake_physics = 1
lake_state_source = "auto"
```

The ERA5-Land row accepts an acquired CF NetCDF file (CC-BY-4.0; the receipt
carries its citation):

```toml
[shared]
sf_lake_physics = 1
lake_state_source = "era5-land"
lake_state_path = "/inputs/lake-fields.nc"
```

The file contains `valid_time`, `latitude`, `longitude`, and the fields `lmlt`,
`lblt`, `lmld`, `lshf`, `licd`, `lict` on those axes, in K, K, m, 1, m, K.
Longitude may be in either conventional encoding. The latest analysis at or
before initialization is selected; future analyses and stale files are refused.
An explicit source refuses missing fields, bad units and invalid physical
values. Automatic acquisition reports partial coverage and errors without
replacing absent values with zero. Each field has a file hash, selected time,
age, URL and coverage count in the initialization receipt.

Outside GLSEA (the Alps, Scandinavia, African and Asian lakes), lakes
the forcing does not resolve start from the forcing's skin temperature, which on
the GFS route is the nearest GFS water: in the Alps, every lake started at 291 K
on 2026-11-09, the Mediterranean's temperature. Two opt-in values replace it
with ERA5's FLake lake state, read keylessly from the public Google ARCO ERA5
mirror:

```toml
[shared]
lake_state_source = "auto+era5"   # GLSEA where it covers, ERA5 elsewhere
# lake_state_source = "era5"      # ERA5 for every lake cell
```

The `era5` row reads the same six FLake fields as the ERA5-Land row, so the
CLM columns get the same profile transfer. It takes the latest hour that the
store has published at or before the start. ERA5T trails real time by about
five days, so a live forecast starts its lakes from a state up to 240 hours
old (persistence); the receipt records that age. FLake runs at every ERA5
point, using that box's GLDB depth, so its 0.25 degree values are lake
states, not land skin temperatures. No height correction is made. ERA5 is a
model, so the forcing's ice stays in charge (`ice_authority = "forcing"`):
- a cell the forcing has frozen (the GFS route's observed IMS lake ice)
  keeps its ice, and takes ERA5's ice thickness and temperature only where
  ERA5 is frozen too;
- a cell the forcing has open never takes ERA5 ice.

`auto+era5` binds GLSEA first and asks ERA5 only for the cells that GLSEA
left without a water temperature. Both values change the initial state
wherever the domain has lakes, so neither is a default.

The ARCO store declares ERA5T further ahead than it has written chunks (on
2026-11-16 it declared 2026-11-10, but its last written hour was
2026-11-08 23 UTC). An hour whose fields come back entirely missing is
therefore skipped for the same hour one day earlier, within the age limit.

Surface water temperature at the 2026-11-09 12 UTC start of the Alps case
(Simstrat is Eawag's operational 1-D lake model, from the Alplakes API. It
is the reference here because no public in-situ lake temperature is
available for these lakes):

| lake | GFS route (default, 1 km cell) | `era5` (2026-11-08 12 UTC) | Simstrat |
|---|---|---|---|
| Geneva (Lausanne) | 18.0 C | 10.7 C | 12.5 C |
| Constance | 12.9 C | 7.4 C | 11.4 C |
| Neuchatel | 17.8 C | 8.9 C | 11.7 C |
| Zurich | 17.8 C | 9.5 C | 12.7 C |
| Maggiore | 18.1 C | 11.7 C | 15.9 C |
| Sils (1797 m) | 12.9 C | 6.5 C | 6.6 C |

Against Simstrat, the default is warm by 1.5 to 6.3 K (mean absolute
error 4.5 K over the six lakes). ERA5 is cold by 1.8 to 4.2 K in the deep
lowland lakes, and within 0.1 K at Sils (mean absolute error 2.7 K). It is
worse than the default at Constance and Maggiore. ERA5's lake depth for the
Constance box is 9 m, so FLake there cools like a shallow lake.

Public ARCO ERA5 atmospheric forcing is available without a CDS key through
`woof fetch --source era5 --era5-provider arco`. ERA5 is a separate product;
it is not evidence that ERA5-Land profile fields have been acquired. The local
ERA5-Land door does not download authenticated CDS files.

Rust reads CF masks/packing, samples finite local bilinear donors and converts
to SI. It does not extrapolate beyond the raster. CUDA binds TSK/SST and the
horizontal ice reservoirs before LSM initialization. With `sf_lake_physics=1`,
it also binds the sparse CLM columns after lakeini:

* GLSEA constrains the surface. Deeper temperature and unobserved ice thickness
  retain lakeini priors. Open-water concentration clears column ice. A GLSEA
  ice fraction alone cannot supply an observed thickness; the receipt counts
  those icy cells explicitly.
* ERA5-Land mixed-layer temperature, bottom temperature, depth and integral
  shape factor reconstruct a power profile. Below the mixed layer, normalized
  depth `r` gives `T = Tmixed + (Tbottom - Tmixed) r^(1/shape - 1)`. Its depth
  mean shape is the supplied factor. This is an explicit transfer between
  lake-model representations, not a measured vertical temperature profile.
  A mixed layer deeper than model bathymetry yields an isothermal column.
* Positive ERA5-Land ice thickness identifies an ice-covered column. When a
  concentration is supplied, thickness times concentration is conserved as
  equivalent ice volume distributed from the top downward. Equivalent ice
  thicker than model bathymetry is refused before binding. Ice temperature
  caps frozen layers at melting. Sediment temperatures remain lakeini priors.

Existing serialized `lake_columns`, `lake_static`, `lake_latitude` and surface
fields retain prognostic state across restart. Source acquisition runs only on
cold start; provenance is run-local metadata and never overwrites a restart.

`WOOF_LAKE_STATE_CACHE` sets the download cache. `WOOF_LAKE_RECEIPT_DIR` sets
the JSON receipt directory. The native preprocess library must be rebuilt from
`tools/grib1_bridge/Cargo.toml`; there is no Python decoder fallback.

Verification helpers decode NDBC WTMP and missing sentinels in Rust, select a
bounded time window, match only nearby finite wet model cells and reduce paired
errors in parallel. Missing buoy reports stay unavailable. GLSEA initialization
agreement measures binding closure, not independent forecast skill. Buoy bulk
water temperatures and model skin temperatures are distinct measurements.
`write_diagnostic_cf` masks land and converts units in Rust, then writes through
the existing Rust NetCDF seam. Render its products using `woof.rustwx` and the
production `rw_wrfbatch`; no alternate weather renderer is used.

The CF renderer table includes lake temperature in degF, lake ice fraction,
accumulated snowfall water equivalent in inches and snow depth in inches.
SNOWNC is water equivalent, not snowfall depth; SNOWH is the current snowpack
depth, not a storm accumulation. The event proof keeps those products separate.
The native renderer can load the checked-in CF table with `--cf-table` when
using an older prebuilt renderer executable.

Regular longitude axes may ascend or descend and use either conventional longitude encoding. Sampling chooses an equivalent coordinate within the actual source extent. In particular, a descending axis crossing zero does not imply a 0-to-360 encoding. No donor is extrapolated beyond a regional source boundary.

The source table may declare `zero_roundoff_tolerance` in SI units. Rust snaps only values whose absolute magnitude is within that tolerance to zero. ERA5-Land ice depth uses 1e-12 m because converted GRIB zero depth can be -2.1684e-19 m. The receipt records the tolerance; material negative depths still fail admission.

## Lake extent

Lake ICs only help where the lake mask is right. The WPS 30-arc-second
lakes include glacier and rock in the Alps, and their shorelines sit about
1 km north of where they are. `[static.highres] fields = "lakes"`
([high-resolution-static-geography.md](high-resolution-static-geography.md))
takes the lake extent from CGLC-MODIS-LCZ instead (opt-in).

## Hex

On hex's WOOF column route (R1), `lake_state_source` is a `--physics` key.
`woof hex forecast --physics sf_lake_physics=1 --physics
lake_state_source=auto+era5 --surface-statics <mesh>.surface_statics.nc`
binds the same rows, through the same `initialize_surface`, on the cells
that the surface statics mark as lakes. The CLM columns that result are WRF
lakeini's columns from that initial state, bit for bit
(`tests/test_hex_surface_inputs_gpu.py`, CUDA and Metal). Without
`sf_lake_physics = 1`, no lake mask is staged on the mesh, so no lake
state is bound either. The MPAS suite route (R2) folds lakes to water, as
MPAS-A v8.4.2 does.
