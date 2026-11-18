# Lakes

None of the lake options are on by default. Each one changes the forecast
wherever the domain has lakes. The default WRF route keeps WRF's lake
treatment: lakes are water points, with the skin temperature the forcing gives
them, and GLSEA (`lake_state_source = "auto"`) wherever it covers.

## Recommended settings

On the WRF route (`[model] engine = "wrf"`), where lakes matter to the
forecast, run with both of these:

```toml
[static.highres]
enabled = true
cache_root = "/data/highres-cache"
fields = "lakes"     # lake extent from 100 m land cover
```

```text
woof fetch-geog --datasets lake_depth --source ncar   # once: WPS bathymetry
woof go forecast.toml --lakes clm
```

They stay opt-in because they change the forecast. `--lakes clm` is refused
on hex; the hex lake routes are under [Hex](#hex). What they did on the
cases we scored:

- **Alps, 1 km, 24 h (25 SMN stations within 3 km of a lake shore and below
  1000 m, CUDA and Metal agree to 0.01 K).**

  | run | shore T2 bias / RMSE (K) | shore 10 m wind bias / RMSE (m/s) |
  |---|---|---|
  | defaults | +1.66 / 2.34 | +1.01 / 1.86 |
  | `--lakes clm` | +0.49 / 1.45 | +0.67 / 1.55 |
  | `--lakes clm` + `fields = "lakes"` | +0.43 / 1.49 | +0.57 / 1.44 |

  `--lakes clm` gives most of the T2 gain. Adding `fields = "lakes"` lowers
  the T2 and wind bias and the wind RMSE further, and moves the T2 RMSE by
  +0.04 K, within the run-to-run noise of this sample. It also corrects the
  lake extent: the 480 WPS lake cells above 2000 m (glacier and rock) became
  8 real reservoirs. Stations far from the lakes and high stations moved by
  less than 0.05 K.
- **Great Lakes lake-effect snow, 3 km, 48 h (Buffalo, 17 Nov 2022, against
  MRMS; `--lakes clm` only, `fields = "lakes"` not scored here).** Here
  `--lakes clm` makes the snow worse: the Erie lee mean rises to 31.9 mm
  against 28.6 mm with the defaults and 27.9 mm observed, and the caveats
  below come from this case. Weigh it before turning on `--lakes clm` for
  lake-effect snow.

Caveats:

- **The Great Lakes cool too fast.** Over 48 h the CLM lake cools Erie by
  3.3 K where GLSEA shows 1.5 K (2.2 times as fast), and Ontario by 2.9 K
  against 1.0 K (2.9 times). This is WRF's CLM lake behaviour: the kernel is
  bitwise WRF's.
- **Lake-effect cores are over-intense.** The 50 mm FSS falls from
  .81/.89/.94 to .70/.82/.88 at 9/27/51 km, and the Erie lee band sits
  8.6 km south of MRMS (7.5 km with the defaults). The 10 mm FSS improves
  (.76/.83/.88 to .79/.86/.90).
- **`lake_depth` is a separate download.** `woof go` does not fetch the WPS
  `lake_depth` geography; stage it with the `woof fetch-geog` line above
  (or see below), or set `use_lakedepth = 0` to run on the uniform
  `lakedepth_default`.
- **Hex R2 starts too cold.** On the hex MPAS-suite route
  (`config_clm_lake_backend`), lakes start from the forcing's skin
  temperature (lakeini), about 1.4 K colder than GLSEA on the Great Lakes
  case; `lake_state_source` is not bound there.

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
