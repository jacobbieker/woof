# The wind-energy model column

A wind-farm reanalysis runs on model levels. The column has to resolve the
rotor layer (roughly 30 to 250 m above ground) and stop at or below 20 km.
`woof.energy.vertical` builds that column and checks it. `woof domain` and
both WRF energy planners write it as explicit `eta_levels` and `p_top` in
every TOML they emit.

## The flags

`woof domain`, `woof energy plan --topology wrf-nests` and
`woof energy plan --topology wrf-tiles` take the same six flags:

| Flag | Meaning |
|---|---|
| `--p-top-pa PA` | Model top pressure in Pa, written as `[shared].p_top` |
| `--eta-levels LIST\|FILE` | Explicit full eta levels from 1 down to 0, strictly decreasing. Give a comma list, or a file of numbers separated by commas, spaces or newlines (`#` starts a comment). `nz` is the number of levels minus one |
| `--dzbot M` | Depth of the lowest layer for WRF `compute_eta` |
| `--dzstretch-s X` | Layer stretch factor near the surface |
| `--dzstretch-u X` | Layer stretch factor aloft |
| `--max-dz M` | Deepest layer allowed |

Any `--dz*` flag generates `nz + 1` levels up to the model top. The
generator is WRF 4.6.1 `compute_eta` with `auto_levels_opt = 2`
(`woof.ingest.eta.generate_wrf_eta`). It needs the native CPU bridge (see
`woof doctor`). `--eta-levels` cannot be combined with the `--dz*` flags,
and `--nz` must agree with an explicit ladder. `hex-swath` refuses all six
flags, because it builds its own MPAS column.

When none of the six flags is given, nothing changes. `woof domain` emits
its 49-level ladder (resampled by `--nz`) under the source-bounded 50 hPa
default. The planners keep their own ladders, and `--nz` alone keeps its
old meaning.

## `woof domain`: a column of your choosing

`woof domain` only checks that the column is valid. It applies no
wind-energy rules:

* the ladder starts at exactly 1.0, ends at exactly 0.0, decreases
  strictly and is finite, with 4 <= nz <= 256 (the vertical stencil and the
  dynamical core's ceiling);
* `--p-top-pa` must lie inside the source's reach. That is the registry
  row's `extendable_source_top_pa` when its fetch can extend upward,
  otherwise its `certified_source_top_pa`. ICON-D2, for example, stops at
  6000 Pa. The preparation's vertical-coverage check (`woof.adapt`) still
  checks the levels actually fetched;
* the physics level caps (`woof.physics_vertical_contract`, 256 for
  YSU/MYJ/KF, the microphysics rows and the radiation above-model layers
  at this `p_top`) are checked by the `woof check` that `woof domain` runs
  on its own output.

The emitted header states the column:

```text
# VERTICAL COLUMN: nz = 70 mass levels, p_top = 5500 Pa, eta_levels wrf-compute-eta
#   WRF 4.6.1 compute_eta controls: auto_levels_opt = 2, max_dz = 1000, dzbot = 12, dzstretch_s = 1.3, dzstretch_u = 1.1
```

`woof domain` also prints one `domain: column ...` line with the top
height and the lowest full levels.

## `woof energy plan`: the wind-energy preset

Any of the six flags on a WRF planner builds
`wind_energy_column(nz, p_top_pa, dzbot_m, ...)`. Unset values take the
preset:

| Setting | Preset | WRF's own default |
|---|---|---|
| `p_top` | 5500 Pa (19.97 km) | 5000 Pa (20.6 km) |
| `nz` | 70 | - |
| `dzbot` | 12 m | 50 m |
| `dzstretch_s` | 1.15 | 1.3 |
| `dzstretch_u` | 1.1 | 1.1 |
| `max_dz` | 1000 m | 1000 m |

On top of the `woof domain` checks, the preset refuses two things:

* **A model top above 20 km.** The height of `p_top` is computed as its
  geopotential altitude in the 1976 standard atmosphere
  (`isa_height_m`). The shallowest top it accepts is 5474.9 Pa
  (`MIN_P_TOP_PA`). WRF's 50 hPa default is about 20.6 km, so it is
  refused:

  ```text
  p_top 5000 Pa is 20.58 km in the standard atmosphere, above the
  wind-energy column's 20 km ceiling; use --p-top-pa >= 5475 (the preset
  default is 5500 Pa)
  ```

* **Fewer than 8 whole layers below 250 m** (`DEFAULT_MIN_ROTOR_LAYERS`,
  `ROTOR_TOP_M`). The preset gives 10, with full levels at about 12, 26,
  41, 59, 80, 104, 131, 162, 198 and 239 m. WRF's own `dzstretch_s = 1.3`
  gives 7, and so does the 49-level default ladder. Both are refused.

Heights are quoted for a reference column: flat ground with the surface
at the WRF reference pressure P0 = 1000 hPa, where
`p = eta (P0 - p_top) + p_top` holds exactly for every `hybrid_opt`. The
standard atmosphere converts pressure to height, measured above that
surface. A real column moves these numbers by a few per cent with
temperature and terrain. Treat them as a design statement, not a
forecast.

The planners place the column as follows:

* **wrf-nests**: every domain of the one configuration carries it. Every
  domain is checked against its physics level rows at this `p_top`
  (`check_physics_levels`). The configuration then loads through the
  same `woof check` gate.
* **wrf-tiles**: the parent is emitted by `woof domain` with `--p-top-pa`
  and an `--eta-levels` file (`parent/eta_levels.txt`). Every leaf tile
  carries the same ladder in its child TOML, and each tile is checked
  against its physics level rows. The parent TOML's header reads
  `eta_levels explicit`, because `woof domain` received the ladder as a
  file. Its provenance is the comment line of `parent/eta_levels.txt` and
  `vertical.json`. Intermediate tiles inherit the parent's
  ladder. The leaf `child_levels` record shows `"stretch": null` and
  `"column": "vertical.json"`.

Both planners write `vertical.json` (`woof-energy.vertical.v1`) next to
`plan.json`. It holds `nz`, `p_top_pa`, `model_top_m`, `eta_levels`, the
reference full- and mass-level heights, `rotor_layers`, the generator
controls and the generator's receipt (bridge sha256 and the sha256 of the
float32 eta). The first domain of the plan has a `vertical` summary in
its `extra` that points at this file, and the planner's JSON summary
includes the same summary.

## Choosing a boundary source

The model top has to lie inside the source's column:

| Source | Reaches | 5500 Pa preset |
|---|---|---|
| `gfs` | 1 Pa (fetch extends upward) | accepted |
| `icon-d2` | 6000 Pa (certified, no extension) | refused; use `--p-top-pa 6000` (19.4 km) |
| `rrfs-ens` | 25000 Pa | refused |
| `20crv3-cf` | 10000 Pa | refused |
| rows with no declared top | checked at preparation | - |

## Example

```bash
woof domain --point 55.4,-3.2 --root-dx 3 --hours 6 --source gfs \
  --cycle 2026-10-01T00 --vram-gib 80 --p-top-pa 5500 --dzbot 12 --nz 70 \
  --out scot.toml
woof check scot.toml
woof energy plan sites.json --topology wrf-tiles --dx-m 100 --source gfs \
  --start 2026-10-01T00 --hours 6 --p-top-pa 5500 --dzbot 12 --vram-gib 80 \
  -o plan-tiles
```

The two commands differ. `woof domain --dzbot 12` uses WRF's other
defaults (`dzstretch_s = 1.3`, 7 layers below 250 m). The planner uses
the preset (`dzstretch_s = 1.15`, 10 layers).
