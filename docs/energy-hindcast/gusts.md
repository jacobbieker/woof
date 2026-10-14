# In-model wind maxima and gusts

A wind-farm reanalysis needs the extremes *between* output frames: the
peak 10 m wind, the strongest gust and the vertical-motion extremes over
each interval, not the instantaneous values at the frame time. WOOF folds
them inside the model on every time step, on the GPU, and publishes them
in each history frame.

Switch them on with `nwp_diagnostics = 1` (`&time_control` in a WRF
namelist, or the RunConfig key of the same name). The `energy`,
`minimal` and `severe` history presets select them. A preset does not
switch the diagnostic on, so an energy run that wants them sets the key.

| variable | units | what each frame carries |
|---|---|---|
| `WSPD10MAX` | m s-1 | maximum 10 m wind speed since the previous frame |
| `U10MAX`, `V10MAX` | m s-1 | the 10 m wind components at the instant of `WSPD10MAX` (its direction) |
| `GUST` | m s-1 | maximum surface gust since the previous frame (method below) |
| `W_UP_MAX` | m s-1 | strongest updraft on any full level with pressure above 400 hPa |
| `W_DN_MAX` | m s-1 | strongest downdraft on the same levels (negative, or 0) |

All six are float32 `(Time, south_north, west_east)` fields with WRF's
attribute conventions. They ride beside `UP_HELI_MAX`, which the same
switch turns on.

## Semantics

These are history-interval running maxima, the WRF `nwp_diagnostics`
semantics (WRF v4.6.1 `phys/module_diag_nwp.F`, `diagnostic_output_nwp`):

- every model step folds the step's value into the running maximum;
- the accumulators are zeroed after each history frame is written, so a
  frame's value covers the interval (previous frame, this frame];
- they are carried in checkpoints, so a resumed run continues the open
  interval rather than restarting it.

Members, as WRF defines them:

- `WSPD10MAX`: `sqrt(u10*u10 + v10*v10)`, kept when it exceeds the
  running value. `u10`/`v10` are the surface-layer scheme's 10 m wind.
- `U10MAX`/`V10MAX`: the components stored when `WSPD10MAX` grows. This
  is WRF's `varstatisticsWIND` rule (`phys/module_diag_cl.F`), not an
  independent maximum of each component, so the pair gives the direction
  of the peak wind.
- `W_UP_MAX`/`W_DN_MAX`: over every full level `k`,
  `IF (p8w(k) > 40000 .AND. w(k) > w_up_max) w_up_max = w(k)`, and the
  mirror with `<` for the downdraft. `p8w` is WRF `phy_prep`'s
  full-level pressure (`dyn_em/module_big_step_utilities_em.F`):
  interior `fnm(k)*p(k) + fnp(k)*p(k-1)`, the surface extrapolated
  linearly in height from the two lowest mass levels, and the top
  log-linearly from the two highest.

## The gust method

`GUST` is the NOAA-EMC Unified Post Processor's `CALGUST.f` estimate, the
branch the RAP, HRRR and FV3-based regional systems use (Manikin 2003 and
later revisions; the operational RAP/HRRR post, Benjamin et al. 2016,
*Mon. Wea. Rev.* 144, 1669-1694). It mixes momentum from inside the
boundary layer down to the surface, reduced with height:

```
SFCWIND = |V10|
gust    = SFCWIND
for every mass level k from the second one up to one level above the PBL top:
    DZ   = height of level k above the ground
    gust = max(gust, SFCWIND + (|V_k| - SFCWIND) * (1 - min(0.5, DZ / 2000 m)))
```

`|V_k|` is the model-level wind speed at the mass point (the staggered
`U`/`V` averaged onto the column). The reduction factor is 1 at the
surface, 0.75 at 500 m and 0.5 from 1 km up, so a strong low-level jet
inside a deep mixed layer contributes half its excess over the 10 m wind.
`GUST` is never below the 10 m wind. WOOF evaluates this every model step
and keeps the interval maximum, the same "maximum since the previous
output" meaning as ECMWF's `10fg`; UPP itself evaluates it only at output
times.

The PBL height is the one the run publishes as `PBLH`: the PBL scheme's
field, or, under SASE (`bl_pbl_physics = 900`), the bulk-Richardson height
the history writer computes for that scheme.

Two choices are WOOF's where the sources leave one:

1. The PBL-top level (UPP's `LPBL`, computed by UPP's caller as "the model
   level nearest the PBL top") is the lowest mass level whose height above
   the ground is at least `PBLH`, or the top mass level when none is.
   The gust therefore always samples every level inside the PBL plus one
   above it. A `PBLH` of 0 still reads the second level, as UPP's loop
   does.
2. Without a surface-layer scheme (no physics driver, so no `u10`, `v10`
   or `PBLH`), `WSPD10MAX`, `U10MAX`, `V10MAX` and `GUST` stay 0. For
   `WSPD10MAX` that is what WRF writes, because its `u10` is zero without
   `sf_sfclay_physics`; for `GUST` it means no gust is invented from a
   missing surface wind. A driver that has a 10 m wind but no PBL height
   field still folds the three 10 m members, and only `GUST` stays 0.
   `W_UP_MAX`/`W_DN_MAX` are always computed.

ECMWF's turbulent gust (`U10 + 7.71 u*` with a stability factor, plus a
convective term from the 850-950 hPa shear) is not implemented. It needs
`u*` and the Obukhov length on every step, and it is an alternative
estimate, not WRF behaviour.

## Where WOOF differs from WRF

1. **Reset instant**, the same as `UP_HELI_MAX`. WRF zeroes the maxima on
   the first step after a history write and then folds that step. WOOF
   zeroes them as soon as the frame is written and folds the end-of-step
   state, so each window includes its own frame's instant. The steps
   inside the interval are the same; one endpoint sample differs.
2. **`p8w` instant.** WRF's 400 hPa test uses the `p8w` that `phy_prep`
   built from the start-of-step state against the end-of-step `w`. WOOF
   evaluates both on the completed step. Only the level mask near
   400 hPa can differ, by one step's pressure change.
3. **The 10 m wind** is the surface-layer output of the step's physics
   call, which is computed from the start-of-step state. That is exactly
   WRF's `grid%u10` at diagnostic time. When `bldt > dt`, both hold the
   last call's value.
4. **`U10MAX`/`V10MAX` window.** WRF has them only in its
   `output_diagnostics` climate package, on the auxhist3 interval and
   seeded from the first sample. WOOF keeps them in the `nwp_diagnostics`
   window beside `WSPD10MAX`, seeded at 0 and reset at history writes, so
   the three always describe the same instant.
5. **Sub-stepping**, the same as `UP_HELI_MAX`. Under a compatibility
   integrator that runs internal substeps, the fold happens on every
   internal step.

## Implementation and checks

- `woof/core/wind_gust_diag.py` owns the fold, the NumPy reference and
  the reset. `woof/core/kernels/wind_gust_diag.cu` runs one CUDA thread
  per column, every float32 operation an explicit rounding intrinsic so
  that it matches the NumPy reference bit for bit. The one exception is
  the model-top `p8w`, which goes through `exp`/`log` and can differ by a
  few ULP. That matters only for a model top within those ULP of 400 hPa.
- Every member is column-local, so tiles, multi-GPU decomposition and
  nests need no extra halo. Each domain owns its accumulators and resets
  them at its own history writes.
- `woof/core/interval_maxima.py` holds the reset registry every history
  publisher calls after the `UP_HELI_MAX` reset.
  `WindMaximaIntervalDiagnostic` exposes the same fold, fields and reset
  as a registry member (`on_step`, `history_fields`, `close_interval`).
- The accumulators are serialized scratch slots
  (`woof/io/restart.py:SERIALIZED_SCRATCH_SLOTS`). A checkpoint written
  before they existed restores them as zero with a note. A checkpoint
  that carries them, resumed with `nwp_diagnostics = 0`, drops them with
  a note.
- `tests/test_wind_gust_diag.py` checks hand-computed columns for the gust,
  `p8w` and the 400 hPa mask. It also covers the `U10MAX`/`V10MAX`
  instant rule, accumulation and reset, restart round trips, emission,
  the `energy` preset, and every publisher's reset. Its GPU tier compares
  the kernel with the NumPy reference bit for bit on flat and terrain
  columns, and runs 60 s of a 9 x 8 x 30 dycore domain with a frame every
  15 s, checking every frame against the reference and the reset between
  frames.
