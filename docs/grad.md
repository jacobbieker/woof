# `woof grad`: observation cost and adjoint-free sensitivities (experimental)

`woof grad` is opt-in tooling that reads finished forecasts. It is phase P0
of [dev/DIFFERENTIABLE-HYBRID-PLAN.md](dev/DIFFERENTIABLE-HYBRID-PLAN.md).
It changes no forecast: nothing on a forecast path imports `woof.grad`
(`tests/test_grad_obs_cost.py` checks the import graph), and the command
is dispatched only through the component door, like `woof hex`. Its file
formats may change in a minor release.

## The observation cost

For a forecast with history frames `x_t`, the cost is

    J_o = 1/2 * sum_i ((H_i(x_t(i)) - y_i) / sigma_i)^2

over the observations `y_i` in a time window. The error matrix R is
diagonal: each observation uses its own reported error when it has one,
and otherwise the error from a type table.

- **Time (FGAT).** Each observation is scored against the history frame
  nearest its time. It is dropped if no frame lies within half the
  history interval (`--max-offset-minutes` changes this). Hourly history
  therefore scores hourly reports exactly and aircraft reports to within
  30 minutes.
- **Columns.** An observation belongs to the model column whose centre is
  nearest, on WRF mass points and hex cells alike. It is dropped when that
  column lies in the WRF boundary rim (`--rim`, 5 by default) or in the
  hex relaxation rings (`bdyMaskCell != 0`), or when the observation is
  outside the domain. The rule is the one the observation referees use
  (`woof.hex.sonde_profile_referee.nearest_cells`).
- **Operators.** Each one reuses woof's existing operator for that
  quantity:

  | kind | operator |
  |---|---|
  | `surface_temperature` | `T2` |
  | `surface_dewpoint` | `Q2` and `PSFC` through Bolton's dewpoint (`woof.obs.column_operators`) |
  | `surface_wind_u`, `_v`, `_speed` | `U10`/`V10`, earth-relative (WRF winds rotated by `SINALPHA`/`COSALPHA`) |
  | `upper_temperature`, `upper_wind_u`, `_v` | linear in ln p between the bracketing model levels; never extrapolated |
  | `refractivity` | p log-linear and T, q linear in height, then the Smith-Weintraub refractivity |
  | `cloud_water_path` | `woof.da.obsop_cwp` on WRF (dry column mass); the sum of `q rho dz` on hex |

  Surface reports are also dropped when the station's elevation differs
  from the model terrain by more than the type's `max_elevation_diff_m`
  (200 m in the default table).
- **Background check.** A paired observation is rejected when
  `|H(x) - y| > gross_sigma * sigma`.
- **Frozen observation set.** Finite-difference sensitivities need the
  same observations in every run, or J jumps when one observation crosses
  the rejection threshold. Score the reference run with `--pairs-out`,
  then score every perturbed run with `--accept` set to that file. The
  background check is then skipped, and only the reference's accepted
  observations are scored. An accepted observation that a perturbed run
  cannot simulate is counted as `lost_from_reference`, never silently
  dropped. If the reference was scored against a different observation
  file, `--accept` refuses.

The default error table is
[`woof/data/grad/obs-cost-types.v1.json`](../woof/data/grad/obs-cost-types.v1.json).
It holds rough values for each platform and kind, and it is
**uncalibrated**. Pass your own table with `--error-table`, in the same
schema (`woof.grad.obs-cost-types/v1`). Each row gives either `error`
(fixed, in the observation's unit) or `relative_error` (a fraction of the
value, interpolated in height, as refractivity uses).

## Observations

`woof grad obs` reads the observations for a forecast's window and domain
from the public bkr/obs Icechunk stores on source.coop. It writes them to
one `.npz` file of plain arrays. Keep that file outside the repository.

| store | rows |
|---|---|
| `wis2_land` | SYNOP 2 m temperature and dewpoint, and 10 m wind (sensor heights 1-3 m and 8-12 m). A corrected report replaces its original. |
| `wis2_upper_air` | radiosonde temperature and wind at the mandatory levels, at each level's own time and drifted position |
| `wis2_aircraft` | aircraft temperature and wind, at the pressure of the reported flight level |
| `metop_sg_ro` | refractivity from Abel-inverted bending angles, thinned to 500 m in height, up to 30 km |

`--goes-cwp FILE` adds cloud water path observations from a
`gpuwm-obs.goes-grid.v1` product, with that product's own errors.

```bash
woof grad obs  --history RUN/ --out /scratch/obs.npz
woof grad cost --history RUN/ --obs /scratch/obs.npz --pairs-out /scratch/ref-pairs.npz --out ref.json
woof grad cost --history PERTURBED/ --obs /scratch/obs.npz --accept /scratch/ref-pairs.npz --out pert.json
```

`--history` takes run directories (it finds `wrfout_d01_*`, or the hex
`*history*.nc`) or the files themselves. The history must carry the full
state; hex histories without `zgrid`/`bdyMaskCell` need `--mesh` to point
at the grid, static or init file. The summary JSON gives `J`, `J` per
observation, and for each type the count, `J`, bias, RMSE and mean sigma.
It also counts every reason an observation was dropped, and records the
SHA-256 of the observation set and of the error table.

## Finite-difference sensitivities

The derivative of `J_o` along a direction `p` in the initial state is

    dJ/d(eps) ~= (J(x0 + eps p) - J(x0 - eps p)) / (2 eps)

Each side is a forecast restarted from a perturbed state and scored with
`--accept`, so both count the same observations.

- **WRF route.** `woof grad perturb` writes `CKPT + eps * p` as a new
  restart checkpoint through the same writer an analysis uses
  (`woof.ensemble.increments.apply_increments_to_checkpoint`), which caps
  vapour at saturation and repairs microphysics number moments. Keep the
  file name in the `gpuwmrst_d01_<time>.npz` form, put it outside the new
  run's output directory, and continue with
  `woof go CFG --prepared-root BASE/prepared --restart FILE`. The restart
  must be run by the same woof source as the base run, and a checkpoint is
  written only when the base run used `--keep-checkpoints`. Pass the base
  run's `--mp-physics`, and `--morr-rimed-ice` for Morrison, so the
  writer's number-moment repair matches the run's microphysics. WRF-route
  blobs are zero in the outer `--rim` rows and columns (default 5, the
  specified and relaxation zone).
- **hex.** Run the base with `--checkpoint-hours H --jedi-background`.
  `woof grad perturb --checkpoint BACKGROUND.nc` writes a copy of the JEDI
  background at hour H with `eps * p` added. Continue with
  `--restart-from CHECKPOINT.pkl --da-analysis COPY.nc`, which rebuilds
  the dycore state from it. Directions on hex are `theta` fields
  `(nCells, nVertLevels)`, and they are zero in the relaxation rings when
  `--mesh` is a static or init file that carries `bdyMaskCell`.
- **The reference.** The writer also repairs number moments the model left
  inconsistent, and those repairs are the same for any `eps`. Score a run
  written with `--scale 0` as `--reference`. Score every run, including
  `--reference` and `--zero`, with the same `--accept` pairs file;
  `woof grad fd` refuses runs scored on other observations. A restart from the untouched
  checkpoint is then `--zero`. On the WRF route, under strict numerics, it
  reproduces the base bit for bit. On hex, the `--da-analysis` restart
  rebuilds the dycore state from float32 fields, so a scale-0 restart moves
  `J` slightly. Use that run, not the base, as the hex reference.

```bash
woof grad direction --checkpoint CKPT --field thp --blob 8 40 50 --radius-cells 5 --out p.npz
woof grad direction --mesh GRID.nc --field theta --at LAT LON 8 --radius-km 30 --out p.npz   # hex
woof grad perturb --checkpoint CKPT --direction p.npz --scale 0.2 --mp-physics 10 --out plus/gpuwmrst_d01_2026-11-07_03_00_00.npz
woof grad fd --leg 0.2 plus.json minus.json --leg 0.1 plus2.json minus2.json --reference zero-scale.json
```

`woof grad fd` gives the central, forward and backward differences for
each scale and the curvature, and how much the derivative changes between
scales. When the summaries carry them, it also gives the central
difference for each observation type, which shows where the response and
the noise come from. The headline `dJ` is taken from the smallest scale
whose derivative is within 25% of the next larger scale's
(`"converged": true`). When no scale meets that, the headline comes from
the largest scale, with `"converged": false`.

Choose `eps` by looking at the scale table. Any perturbation, however
small, moves a 9 h forecast's `J` by a few units, because convection and
QC thresholds respond nonlinearly. On a 9 km regional case, a 1 K `thp`
blob gave a derivative that held steady only for amplitudes of about
0.4 K and above. Below that, the noise dominated.

## Limits

- Radio occultation in the store is bending angle, which is converted
  here to refractivity by an Abel inversion with no ionospheric or
  super-refraction handling. Near the surface, refractivity is the
  weakest part of the profile; the default table's relative error is
  largest there (2 %).
- The GOES stores on bkr/obs hold radiances and cover the Americas only.
  Cloud water path needs a goes-grid product made elsewhere.
- The errors are not tuned per station or per level, and there is no bias
  correction or thinning beyond what is described above.
