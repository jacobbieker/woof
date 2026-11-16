# Sub-grid terrain drag

These optional WRF v4.7.1 schemes use terrain statistics below the model
grid scale. Both selectors default to 0. Turning them on changes the
forecast; column agreement with WRF does not establish forecast skill.

| Selector | Values | Behavior |
|---|---|---|
| `topo_wind` | 0 off, 1, 2 | Corrects YSU surface momentum drag and 10 m wind using terrain variance and terrain curvature. Option 1 uses `VAR_SSO`; option 2 uses `VAR`. Requires `bl_pbl_physics = 1`. |
| `gwd_opt` | 0 off, 1, 3 | Adds orographic drag after the PBL scheme at its cadence. Option 1 provides gravity-wave drag and blocking; option 3 provides large-scale gravity-wave drag, blocking, small-scale gravity-wave drag and turbulent orographic form drag. Requires an active PBL scheme. |

GSL components taper with grid spacing as in WRF. Its large-scale drag
vanishes at 3 km and its small-scale wave and form drag vanish at 1 km.
The port uses WRF's default `gsl_diss_ht_opt = 0`, with no dissipative
heating. `gwd_opt = 2` is unavailable.

Both drag kernels hold a column's stress profile per thread, so a run with
`gwd_opt` 1 or 3 is limited to `nz <= 256`, the dynamical core's own
ceiling. `woof check` refuses a deeper grid by name before preparation
(`TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS` in
`woof/physics_vertical_contract.py`). At 256 levels the option-3 kernel's
per-thread frame is 2,208 B (1,152 B for option 1), measured on sm_86;
the part above the 1,024 B default stack is priced by the VRAM preflight.

With SASE, GSL uses SASE's bulk-Richardson PBL height and its corresponding
upper-bracket level. This BL definition differs from a WRF PBL package.
The diagnostic supplies drag locally and leaves the surface-layer PBL
height and top-level carriers unchanged.

Select an option before preparing the run so the static builder includes
its fields. For example, in the experiment TOML:

```toml
[shared]
bl_pbl_physics = 1
sf_sfclay_physics = 1
topo_wind = 1
gwd_opt = 0
```

The GEOG root must contain the selected WPS datasets:

| Option | Default dataset | Produced fields |
|---|---|---|
| `topo_wind = 1` | `varsso_10m` | `VAR_SSO` |
| `topo_wind = 2` | `orogwd_10m/var` | `VAR` |
| `gwd_opt = 1` | `orogwd_10m` | `VAR`, `CON`, `OA1` to `OA4`, `OL1` to `OL4` |
| `gwd_opt = 3` | `orogwd3_10m` | `VAR`, `CON`, `OA1` to `OA4`, `OL1` to `OL4`, each with `LS` and `SS` suffixes |

Unpack the WPS orographic datasets beneath the same GEOG root used for
terrain and land use. `geog_data_res` in `namelist.wps` selects the WPS
table's resolution alternatives. Missing requested fields stop preparation
with the dataset name; they are never replaced with zero drag.

```sh
woof prep --source SOURCE --input-list inputs.txt \
  --experiment-config experiment.toml --wps-namelist namelist.wps \
  --geog-root WPS_GEOG --output-root prep
woof sim prep --experiment-config experiment.toml \
  --wps-namelist namelist.wps --outdir forecast
```

Replace `SOURCE` with the forcing source and include its required
supplementary inputs, as for a run without terrain drag.

The current routes support a prepared resident root domain, a root split
across cards through `[devices]`, a streamed `[tiles]` root, and a separate
downscale of saved history on its own geography. Each tile receives its
static fields with the same halo and edge mapping as its initial state.
Topographic wind curvature uses the neighboring domain terrain across
stored row slabs. The downscale builds the
requested statistics from WPS_GEOG and passes them to child physics.
Parent-terrain downscales and children inside a running domain tree are
refused because those initializers do not supply the required statistics.
`topo_wind` with BEP or BEP+BEM is refused because
that YSU arm does not contain the surface momentum term it would scale.

## `topo_wind = 1` on hex meshes: measured, not recommended (2026-11-16)

The hex generic column seam accepts `topo_wind = 1` with a `VAR_SSO`
surface-statics file. On the 120 km global mesh it makes the 10 m wind
worse, and it can make the run unstable. Do not use it there.

**The measurement.**

- Setup: four 72 h `global-120km --physics-row arwen` cases, 2026-10-28 to
  2026-11-02. Both arms ran `gwd_opt = 0`, because the WOOF arm cannot yet
  read `VAR_SSO` and the YSU drag statistics from one file.
- Scoring: `woof.hex.station_wind_referee` against WIS2 land SYNOP,
  globally, about 72,000 pairs per case.
- Result: `topo_wind = 1` raises the 10 m wind bias by +0.52 m/s at 24 h
  (95 % case-paired interval +0.43 to +0.60) and the RMSE by +0.87 m/s.
  At 48 h the bias rises by +0.58 m/s and the RMSE by +1.00 m/s.
- The baseline bias is +1.48 m/s at 24 h, so the option takes it to about
  +2.0 m/s.
- Two of the four runs stopped on the dynamical core's validation flag, at
  34 h and 54 h. The two arms without `topo_wind` completed all four cases.

For comparison, the YSU orographic drag (`gwd_opt = 1`, which the
`global-120km --physics-row arwen` suite runs) lowers the bias by 0.23 m/s
at 24 h and 0.31 m/s at 48 h.

**Why.** WRF forms `LAP_HGT` from the four grid neighbours, in metres, and
has no grid-spacing scale (`start_em.F:1558`). It then sets `CTOPO = 0` and
`CTOPO2 = 0` where `LAP_HGT < -30` m. In YSU, `CTOPO` scales the surface
momentum friction (`bl_ysu.F90:1308-1316`), so zero means a frictionless
surface. The thresholds suit grids of a few kilometres. At 120 km on
x1.40962:

- 40 % of land cells fell to `CTOPO = 0` in the measured runs. Those runs
  formed the Laplacian from cell-index neighbours (see below).
- Their 72 h 10 m wind is 3.0 m/s faster than without the option.
- Where `CTOPO > 1` the wind slows by 0.4 m/s, as intended.

**The Laplacian now uses the mesh's own neighbours (2026-11-15).** The
seam lays the columns out as one row, so WRF's kernel used to take i-1 and
i+1 in cell-index order, which are not neighbours on the mesh. Hex now
forms `LAP_HGT` from `cellsOnCell` with `topo_wind_static_mesh` in
`woof/core/kernels/terrain_drag.cu`: the mean of the cell's edge
neighbours minus its own height, in metres, with WRF's -10, -20 and -30 m
thresholds unchanged.

- On a quad neighbourhood listed (i+1, i-1, j+1, j-1) this is
  `start_em.F:1558` word for word. The WRF 4.7.1 oracle fixture checks
  `LAP_HGT`, `CTOPO` and `CTOPO2` bit for bit through the mesh kernel
  (`tests/test_terrain_drag_wrf471_parity.py`).
- On a regular hexagon of spacing dc and a paraboloid h = a r², both forms
  give a dc². The -30 m threshold therefore means the same curvature at the
  same spacing.
- A neighbour outside the column set (beyond a regional boundary) counts as
  the cell itself, which is WRF's clamp at a non-periodic edge.
- The generic column seam refuses `topo_wind = 1` without the neighbours.

The mesh neighbours do not rescue 120 km. On x1.40962, 3,674 of 11,492 land
cells (32.0 %) fall to `CTOPO = 0`, against 39.8 % with index-order
neighbours. The thresholds are in metres and do not scale with grid
spacing, so the option stays selectable, as in WRF, but the seam warns
whenever the mesh spacing exceeds 4 km. The warning gives the run's own
zeroed fraction. The mesh form is the one to use on kilometre-scale
regional hex meshes.

The reproducible column authority is
`tools/terrain_drag_wrf471_oracle/build.sh`; its WRF sources and fixture
files are pinned by SHA-256. The comparison gates are
`tests/test_terrain_drag_wrf471_parity.py`, and option/import/default
identity gates are `tests/test_terrain_drag_config.py`.
Static-field and halo delivery to device ranks is covered by
`tests/test_terrain_drag_devices.py`; device decomposition comparisons are
in `tests/test_terrain_drag_devices_gpu.py`. Rank delivery does not change
the WRF-derived drag routines or add a new Fortran transcription. Their
notices remain in `NOTICE` and `licenses/LICENSE-WRF-public-domain.txt`.
