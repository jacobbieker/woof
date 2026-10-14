# Hub-height wind fields and interval statistics

A wind-farm reanalysis needs the wind at hub height. A history file that
only holds the state at its valid time misses the events that matter most
for a farm, such as ramps, gust fronts and excursions past cut-out, when
they happen between two files. The model therefore computes hub-height
wind after every step and keeps a running record from one history frame
to the next.

Source: `woof/core/hub_wind_diag.py` (the NumPy reference, the inline CUDA
kernel and the diagnostic), and `woof/core/interval_diagnostics.py` (the
small registry that steps it, adds it to frames and resets it).

## Configuration

```toml
[output]
preset = "energy"
hub_heights_m = [40, 80, 100, 120, 150, 200]   # or: hub_wind = true
hub_sites = "sites.json"                       # optional: sites.v1, or inline tables
hub_sample_interval_s = 60                     # optional, default 60
```

| key | default | rule |
|---|---|---|
| `hub_wind` | off | `true` turns the diagnostic on at the default heights. `false` together with any other `hub_*` key is refused. |
| `hub_heights_m` | `[40, 80, 100, 120, 150, 200]` | Strictly increasing, 1 to 32 heights, each in (0, 2000] m above ground. Setting it turns the diagnostic on. |
| `hub_sites` | unset | A `woof-energy.sites.v1` JSON file, resolved from the config's directory and then the working directory, or a list of `{ site_id, lat, lon }` tables. Site IDs must be unique. |
| `hub_sample_interval_s` | `60` | Positive. Refused without `hub_sites`. |

The keys belong to `[output]`, either for the whole tree or on one
domain with `output = { ... }`. A domain's own table replaces the
tree-wide one as a whole. Like the rest of `[output]`, these keys do not
enter the restart identity or the prepared cache. A selection that would
drop every hub field (for example `history_vars = ["T2"]`) is refused,
because the diagnostic would run every step and write nothing. The
`energy`, `minimal`, `severe` and `windfarm` presets keep the hub fields.
`windfarm` also keeps the hub-height scalars below and nothing 3-D, and
is refused without `hub_wind` or `hub_heights_m`.

## Output

The gridded fields share one axis, `hub_height`, which is WRF's approach
for its height-level diagnostics (`z_lev_diags` on `num_z_levels_stag`).
Using one variable per field, rather than one per height, keeps the names
fixed whatever heights a run sets, so they are part of the `[output]`
vocabulary. The writer assigns these fields to the axis by name, so a run
whose number of heights equals `nz` still never writes them on
`bottom_top`.

| variable | dimensions | units |
|---|---|---|
| `HUB_HEIGHT` | `Time, hub_height` | m |
| `HUB_WINDOW_S` | `Time` | s |
| `WSPD_HUB`, `WSPD_HUB_MAX`, `WSPD_HUB_MIN`, `WSPD_HUB_MEAN` | `Time, hub_height, south_north, west_east` | m s-1 |
| `WDIR_HUB` | same | degree |
| `T_HUB` | same | K |
| `P_HUB` | same | Pa |
| `RHO_HUB` | same | kg m-3 |
| `QV_HUB`, `QC_HUB`, `QR_HUB` | same | kg kg-1 |
| `QNC_HUB` | same | kg-1 |

The last five rows are the hub-height scalars. They are written only when
the selection names them (the `windfarm` preset, or `history_vars`), and
`QV_HUB`/`QC_HUB`/`QR_HUB` only on a moist run, `QNC_HUB` only where the
microphysics carries a cloud droplet number (`nc`: Thompson mp 28,
Morrison, Milbrandt-Yau, WDM6). A member the run does not carry is
absent, never zero.

## Arithmetic

1. Mass-level heights above ground for each column:
   `z_w = (PH + PHB) / g` on w levels, then `z = 0.5 (z_w[k] + z_w[k+1]) - HGT`.
2. C-grid destaggering to the mass point at each level:
   `u = 0.5 (U[i] + U[i+1])` and `v = 0.5 (V[j] + V[j+1])`.
3. Linear interpolation in z above ground between the two mass levels
   that bracket each height. A height below the lowest mass level takes
   that level's wind, and a height above the top takes the top's. No
   log-law extrapolation is done, so the model levels should put the
   first mass level below the lowest hub height.
4. Speed is `sqrt(u_h^2 + v_h^2)`, from the grid-relative components.
5. The earth-relative components are
   `ue = u_h COSALPHA - v_h SINALPHA` and `ve = v_h COSALPHA + u_h SINALPHA`.
   Direction is `180 + atan2(ue, ve) * 180/pi`, wrapped to [0, 360). It is
   the bearing the wind blows from, clockwise from north. A calm reports 0.

The scalars use the same heights, the same bracketing and the same
hold rule:

6. `theta = T' + theta_base` per mass level; theta, pressure, `qv`, `qc`,
   `qr` and `nc` are interpolated linear in z between the bracketing mass
   levels (held below the lowest and above the top level).
7. `T_HUB = theta_h (p_h / 1e5)^(R_d/c_p)` and
   `RHO_HUB = p_h / (R_d (T_HUB (1 + 0.608 qv_h / (1 + qv_h))))`, with
   WRF's `R_d = 287` and `c_p = 7 R_d / 2`. A dry run uses `qv = 0`.

This is what `woof energy extract` does with the 3-D volumes (theta,
pressure and vapour sampled at the height, then `air_temperature` and
`air_density`). On one tiny moist GPU run the forecast extracted from the
hub planes and the one extracted from the same frames' volumes agreed to
1.4e-7 relative in wind speed, direction, temperature, pressure, density
and cloud water.

**When the scalars are evaluated.** At the history frame, from the state
the frame is written from, and not in the step epilogue. The dycore's
microphysics adjustment and its pressure diagnosis run after the
epilogue, so an epilogue value would describe a state the frame does not
hold. The scalars are instantaneous: there is no window to restart and
nothing is lost on a resume.

The CUDA kernel uses one thread per column and is compiled with
`--fmad=false`. Its speeds match the NumPy reference exactly, and its
directions differ only by the rounding error of `atan2f`
(`tests/test_hub_wind_diag_gpu.py`). Its scalar kernel
(`hub_scalar_columns`, in the same source) matches the reference exactly
in `P_HUB`, `QV_HUB`, `QC_HUB`, `QR_HUB` and `QNC_HUB`; `T_HUB` and
`RHO_HUB` go through `powf` and differ by at most 1 and 2 ulp
respectively, on about 0.15% of the values (measured on an RTX A6000; the
test grades them at `rtol = 2e-6`). The kernel source is inline and its
SHA-256 is pinned in `tests/test_hub_wind_diag.py`. It is not a
`woof/core/kernels/*.cu` file because those must have a measured
local-memory frame for every per-architecture recording, and this kernel
uses no local memory.

## The window

- **Update.** The diagnostic is updated in the dycore epilogue after
  every completed step (`woof.core.dycore.step`), at the same point as
  `UP_HELI_MAX`. Each step is weighted by its own `dt`, so
  `MEAN = sum(speed dt) / sum(dt)` stays correct under adaptive time steps
  and substeps.
- **Reset.** The window restarts once per published frame, immediately
  after the frame is captured. For the tree runner and both prepared
  runners this happens in `PerDomainWrfoutWriters.submit`. The
  single-domain experiment road uses `write_case_output`, and the offline
  child uses `emit_output`. For the asynchronous writer, the restart
  waits behind the stream event that orders the device-to-host copy, so
  the copy always sees the finished window.
- **Covered interval.** Each window is `(T_previous_frame, T_frame]`.
  When no step has run yet (the cold-start frame), MAX, MIN and MEAN all
  equal the speed at that instant and `HUB_WINDOW_S` is 0.
- **Restarts.** The window is not stored in checkpoints because it is
  output, not model state. After a resume the window starts at the
  restart time, so the first frame covers `(T_restart, T_frame]` and
  `HUB_WINDOW_S` shows this. When the restart interval is a multiple of
  the history interval, the usual setup, every window matches an
  uninterrupted run.
- **No effect on the forecast.** The diagnostic reads `u`, `v`, `ph`,
  `phb`, `ht`, `sina` and `cosa` and writes only its own buffers. On a
  card, `tests/test_hub_wind_diag_gpu.py` steps a domain with and without
  the window and checks that every prognostic field is identical to the
  byte.

## Site sampling

Each site is sampled at its nearest mass cell, found by great-circle
distance. At 100-300 m grid spacing this cell is within about 0.7 dx of
the turbine. A site whose nearest cell is on an outermost row or column,
or more than `max(dx, dy)` from that cell, is outside the domain. Such
sites are listed in the sidecar's `sites_outside_domain` attribute and are
not sampled. In a nested tree, each domain samples the sites that fall
inside it.

Samples are taken at the end of the step nearest each multiple of
`hub_sample_interval_s`, counting from the start of the window, and held
in device memory. Each history write publishes one file atomically:
`hubwind_d0N_YYYY-MM-DD_HH:MM:SS.nc`, with dimensions
`time x site x height`. The file contains:

- variables `time` (seconds since 1970), `height`, `site_id`,
  `site_lat`/`site_lon`, `grid_j`/`grid_i` (0-based),
  `grid_lat`/`grid_lon`, `site_distance_m`, `WSPD` and `WDIR`;
- global attributes `window_seconds`, `sample_interval_s` and the method.

The cold-start frame's sidecar holds one sample, taken at that instant.
If a run crashes, the samples taken since the last frame are lost with
their window.

## Not supported, refused by name

`HubWindUnsupportedRoad` is raised, and the domain is not silently
skipped, in these cases:

- `[tiles]`-streamed domains. Their step is a tile sweep that never
  reaches the dycore epilogue, and their frame is built from the host
  store.
- Ensemble member capture.
- Nest relocation (`refresh_domain`). This check reads the domain's
  `[output]` selection, because by the time it runs the domain's state
  has already been replaced.
- A domain whose state is replaced after its window has taken in steps.
  The window would restart partway through an interval.

When a delayed-start nest is rebuilt at activation, its new state has not
taken any steps yet. The writer reattaches the window to that state at the
first frame, so nothing is lost.

Native batched ensembles report the diagnostic as a reason they are not
eligible and fall back to the original runner.

## Memory and disk

Each domain holds `6 x n_hub` float32 planes on the device, plus the site
buffer, and up to `7 x n_hub` more when the scalars are on. With 6 heights
on a 1500 x 1500 domain that is 324 MB, or 702 MB with every scalar. These
buffers are not scratch slots, so the preflight VRAM plan does not count
them. The disk planner (`history_frame_bytes`) does count the hub planes
the selection writes.
