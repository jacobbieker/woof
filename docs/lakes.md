# Lakes

None of the lake options are on by default. Each one changes the forecast
wherever the domain has lakes. The default WRF route keeps WRF's lake
treatment: lakes are water points, with the skin temperature the forcing gives
them, and GLSEA (`lake_state_source = "auto"`) wherever it covers.

## `woof go --lakes`

| switch | `sf_lake_physics` | `use_lakedepth` | `lake_state_source` |
|---|---|---|---|
| `--lakes clm` (recommended where lakes matter) | 1 | (unchanged, default 1) | `auto+era5` |
| `--lakes ic` | 0 | (unchanged) | `auto+era5` |
| `--lakes off` | 0 | (unchanged) | `auto` |

- `clm` turns on WRF's CLM lake model. It uses WPS bathymetry (the
  `lake_depth` geography) and starts the lakes from GLSEA where GLSEA covers
  and from ERA5's FLake state elsewhere. `woof go` does not download
  `lake_depth` itself (about 2.4 MB to download, about 1.9 GB unpacked; NCAR
  is its only source). Where it is missing, preparation stops and names the
  remedy: `woof fetch-geog --config <stem>.lakes-clm-<hash>.toml` (the copy
  the switch wrote; add `--geog-root` for a non-default geography tree), then
  run `woof go --lakes clm` again. `use_lakedepth = 0` (uniform
  `lakedepth_default`) needs no bathymetry. `clm` also needs an active
  land-surface scheme (`sf_surface_physics` 1, 2, 3, 4 or 8) and is refused
  up front otherwise.
- `ic` keeps WRF's water-point lakes but replaces the forcing's lake skin
  temperature with the same initial state.
- `off` writes the defaults.

Every domain gets the same setting. The switch removes the per-domain `sf_lake_physics` and
`lake_state_*` keys (and `lake_state_path`/`lake_state_accept_licence`);
`use_lakedepth`, `lakedepth_default` and `lake_min_elev` stay as you set them,
per domain or in `[shared]`. `--lakes` is refused with `--branch-from` and on
stage-only launches. Like `--wind-farms`, it
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
