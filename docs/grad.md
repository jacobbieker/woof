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
