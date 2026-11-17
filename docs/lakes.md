# Lakes

None of the lake options are on by default. Each one changes the forecast
wherever the domain has lakes. The default WRF route keeps WRF's lake
treatment: lakes are water points, with the skin temperature the forcing gives
them, and GLSEA (`lake_state_source = "auto"`) wherever it covers.

## `woof go --lakes`

| switch | `sf_lake_physics` | `use_lakedepth` | `lake_state_source` |
|---|---|---|---|
| `--lakes clm` (recommended where lakes matter) | 1 | 1 | `auto+era5` |
| `--lakes ic` | 0 | (unchanged) | `auto+era5` |
| `--lakes off` | 0 | (unchanged) | `auto` |

- `clm` turns on WRF's CLM lake model. It uses WPS bathymetry (the
  `lake_depth` geography) and starts the lakes from GLSEA where GLSEA covers
  and from ERA5's FLake state elsewhere. `woof go` does not download
  `lake_depth` (about 1.9 GB) itself. Where it is missing, preparation stops
  and names the remedy: `woof fetch-geog --config <stem>.lakes-clm-<hash>.toml`
  (the copy the switch wrote), then run `woof go --lakes clm` again.
- `ic` keeps WRF's water-point lakes but replaces the forcing's lake skin
  temperature with the same initial state.
- `off` writes the defaults.

Every domain gets the same setting. The switch removes per-domain lake keys
and `lake_state_path`/`lake_state_accept_licence`. Like `--wind-farms`, it
writes an immutable copy of the configuration beside the original, named
`<stem>.lakes-<mode>-<hash>.toml`, and launches that copy. The switch only
configures the WRF route (`[model] engine = "wrf"`). On a hex configuration it
refuses `clm` and `ic`, and `woof go` on hex refuses `--lakes` by name.

The sources, their licences, the ARCO publication lag and the Alpine lake
evidence are in [lake-state-initialization.md](lake-state-initialization.md).

## Hex

On the WOOF column route (R1), `woof hex forecast --physics sf_lake_physics=1`
runs the same CLM lake kernel. It reads the lake mask and depth from
`--surface-statics` (`woof hex surface-statics`). Every lake call has been
replayed bitwise against WRF's kernel on CUDA and Metal.

## Lake extent

`--lakes` does not change which cells are lakes. Where the WPS 30-arc-second
lake mask is wrong (in the Alps it marks glacier and rock as lakes),
`[static.highres] fields = "lakes"` takes the lake extent from the
CGLC-MODIS-LCZ 100 m land cover (opt-in,
[high-resolution-static-geography.md](high-resolution-static-geography.md)).
