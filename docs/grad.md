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

## Frozen McICA seeds (`--mcica-seed-mode cell`)

RRTMG's McICA cloud generator seeds each column from the fractional digits
of its bottom four layer pressures, `(p_hPa - int(p_hPa)) * 1e9`. That is
upstream behaviour (MPAS `module_ra_rrtmg_lw.F:2432-2440` and the SW
equivalent, WRF's `generate_stochastic_clouds`), and woof keeps it by
default. A one-ulp change in pressure, such as any perturbation of the
initial state makes, therefore draws a new sub-grid cloud realisation in
every cloudy column. On hex, a 3e-5 K theta change (one float32 ulp) moved
`J` by +6.1 and -5.0, whatever its sign. That noise is independent of
the perturbation amplitude, so it sets the floor of every finite-difference
and ensemble-sensitivity leg.

For those legs only, `--mcica-seed-mode cell` replaces the seeds with a
fixed hash of the column id, the radiation call time (model seconds) and
the stream (LW, SW, or RRTMG-K), so the cloud draws no longer depend on
the model state at all. Each seed lies in [1, 999999998], inside the range
the pressure rule produces. The mode is not upstream physics, and it is not
the default; no forecast takes this path unless asked.

| route | how to ask |
|---|---|
| hex | `woof hex ... --forecast-arg=--mcica-seed-mode=cell` (or `--mcica-seed-mode cell` on the forecast door) |
| WRF | `woof go CONFIG --mcica-seed-mode cell` |
| either | `WOOF_MCICA_SEED_MODE=cell` in the environment |

Rules for using it:

- Run the base, the reference and every plus/minus leg in the same mode,
  and with the same column order and time steps. The seeds depend on the
  column's index in the radiation call (the mesh cell on hex, the flattened
  (j, i) point on WRF) and on the call time, not on the state.
- It covers RRTMG (`ra_lw_physics = ra_sw_physics = 4`), RRTMG-K (14) and
  hex's RRTMG LW/SW. RRTMGP (24) refuses it rather than seeding from
  pressure.
- It removes the seed noise only. Convection and QC thresholds still
  respond nonlinearly (see the scale table above).

Demonstration on hex (12 km regional case around Edinburgh, `mesoscale_reference`,
GFS 2026-11-07 00Z, restart at 03 Z, J_o over 07-12 Z, 4,965 observations, a 60 km
theta blob at level 8 with 879 cells, CUDA fast numerics on an A6000). Base, scale-0
reference and every leg ran with `--mcica-seed-mode cell`, and every leg was scored
with the base's accepted observations:

| leg | J | J - base |
|---|---|---|
| base and scale 0 | 1637.976 | 0 (bit for bit) |
| +0.1 K | 1632.734 | -5.241 |
| -0.1 K | 1642.338 | +4.362 |
| +0.5 K | 1633.002 | -4.974 |
| -0.5 K | 1640.774 | +2.798 |
| +1 ulp of theta (3e-5 K) on the 879 cells | 1638.017 | +0.041 |
| -1 ulp of theta on the 879 cells | 1638.019 | +0.043 |
| +1 ulp in one word | 1637.817 | -0.159 |

The +-0.1 K pair is now antisymmetric (central difference -48.0 J/K, symmetric
residue -0.44 J), and the one-ulp legs move `J` by at most 0.16 J. With upstream
seeds, the same ulp legs moved `J` by +6.1, -5.0 and -0.8. The +-0.5 K
pair (central difference -7.8 J/K) still differs from the +-0.1 K pair: what remains
is the forecast's own nonlinearity, not seed noise.

## Ensemble sensitivity and EFSO

`woof grad members` draws K centred initial perturbations from
`woof.da.perturb`. On the WRF route these are a Gaussian random `thp`
field and a non-divergent wind, tapered at the rim. On hex they are
`theta` only, drawn on a tangent plane and sampled at the cells. Each
member is a direction applied with `--scale 1`. On a 9 km WRF case,
`--wind-ms 0` (`thp` only) gave ES that finite differences confirmed in
4 of 6 tropospheric directions, and the default wind perturbation in 1 of
6 (plan, section 5, P0.5).

`woof grad ens` regresses the members' costs on their states:

    S_j = cov(J, x_j) / var(x_j)

It writes the maps of `S`, the correlation and the spread, and lists the
strongest points. `--regress FIELD INDEX` also writes the localised
regression of the whole state on one point, `cov(X, x_j) / var(x_j)`. For
that direction the ensemble predicts `dJ/d(eps) = S_j`, and a finite
difference along it tests the prediction (Ancell & Hakim 2007).

EFSO (Kalnay et al. 2012; Ota et al. 2013) uses `J` as its measure of
forecast error:

1. `woof grad letkf` analyses member checkpoints with the observations
   that a pairs file accepted at the analysis time. It uses
   `woof.da.letkf`, with point batches and winds destaggered and
   restaggered. It writes the analysis members, and it also keeps `Y^a`,
   the analysis ensemble in observation space. `Y^a` comes from the
   filter's own local weights, by carrying `H(x_k)` as an extra analysed
   field.
   - Each `--history` file must hold the analysis time only.
   - Column-integrated observations have no model level, so they are
     left out of the LETKF.
   - The analysis is not relaxed (no RTPS), because EFSO needs the
     Kalman gain's own analysis perturbations.
   - Every member is staged before any is published.
   - `--solve cuda` runs the local solves on the GPU while the state
     stays on the host. On a 100x80x49 grid with 24 members and 729
     observations, this took 43 s; the host solve had not finished after
     70 min. `letkf.json` records the device and eigensolver used.
   - Observations that share a grid point need one extra analysed field
     each: a station's five surface kinds, or a sounding's levels that
     fall on one model level. Each extra field is the size of the state,
     so memory and run time grow with that count.
2. Forecasts from the analysis members are scored with `--accept` and
   `--pairs-out`.
3. `woof grad efso` gives each observation's first-order impact
   `d_i / sigma_i^2 * 1/(K-1) * sum_k Y^a'_ik J'_k`, where
   `d = y - H(mean background)`. Negative values lowered the forecast's
   cost. With `--localization-km`, each verifying observation's share of
   `J` is weighted by its distance from the assimilated one.
4. `woof grad mean` writes ensemble-mean checkpoints. The forecasts from
   the background mean and the analysis mean then give the actual change
   that the summed impact predicts.

## The tensor bridge

`woof.grad.bridge` (plan P1.1) hands woof's device arrays to PyTorch and
back. It is a library module for gradient and NN code; no forecast imports
it, and torch stays an optional extra (`recast-woof[nn-torch]`).

| woof array | `to_torch` | `from_torch` |
|---|---|---|
| CuPy (CUDA) | DLPack, no copy | DLPack, no copy |
| facade (Metal, MLX) | DLPack to `mps`, no copy | one device copy |
| NumPy | `from_numpy` on `cpu`, no copy | `.numpy()`, no copy |
| host facade (CPU MLX) | `from_numpy` on `cpu`, no copy | one copy into the facade |

- **Ordering, CUDA.** `to_torch` makes torch's current stream wait for
  CuPy's current stream with a device-side event, and `from_torch` does
  the reverse, so neither side reads a buffer the other is still writing.
  `woof_stream()` runs torch's work on CuPy's stream instead. A tensor
  used later on yet another torch stream needs torch's own `wait_stream`.
  After torch has *used* a shared view (read it, not only written it),
  call `release` before woof's next launch writes that buffer.
  `from_torch` records the tensor on CuPy's stream, so torch's caching
  allocator does not reuse the memory while woof's queued work reads it.
- **Ordering, Metal.** MLX and torch-MPS use separate command queues with
  no shared event. `to_torch` evaluates the array and drains the facade
  first, and `from_torch` (or `release`) drains torch's queue. Without the
  drain, MLX read stale words of a recycled buffer in a probe.
- **Metal is zero-copy in one direction.** torch-MPS wraps the MLX buffer
  (MLX 0.32, torch 2.10): a write through the tensor is read back by woof,
  for whole arrays and for views. On the way back the bridge asks MLX for
  an explicit copy (`mx.from_dlpack(copy=True)`, evaluated at once), so
  it costs one device copy. float64 copies through the host keep their
  values: the facade stores float64 as bit words, and the bridge decodes
  them.
- **Refusals.** Anything else raises `BridgeRefused` unless `copy=True`
  names the copy: a CuPy array as a CPU or MPS tensor, a tensor on another
  device or card, a facade array as a CPU tensor, a non-contiguous host
  array. float64 cannot go to MPS (torch-MPS has none; use the CPU).
  dtypes are float32, float64, int32 and int64.
- **Autograd.** woof's kernels write in place, and torch cannot see a write
  made by CuPy or MLX. The bridge cannot prevent a shared view from
  entering a graph, so it gives the tools and a rule. `grad_leaf` returns
  an out-of-place copy as a graph leaf: start gradient work from it.
  `from_torch` refuses a tensor that requires grad (detach it on purpose).
  The rule: after woof writes a buffer that a torch tensor shares (a
  `to_torch` view or a `from_torch` result), call `mark_written` on that
  tensor. It bumps the version counter, so a backward that saved the
  tensor raises an error instead of using the new words.

`woof.nn`'s PyTorch runner uses the same bridge, so a TorchScript or
`torch.export` column model also runs on Metal (`backend = "torch"`,
torch-MPS). `tests/test_grad_bridge.py` checks that memory is shared (a
pointer on CUDA and the host; on Metal, a write through torch read back
through MLX), the refusals, and the stream handoff.

## A differentiable scheme: Kessler

`woof.grad.kessler` ports WRF's Kessler scheme to PyTorch for a column
batch. `woof.grad.checks` supplies the dot-product (adjoint) test,
tangent-linear convergence and a per-column `gradcheck`.
`python tools/grad_kessler_g1.py` runs all of them against the WRF oracle
fixture and against woof's own kernel on this host's backend, and prints
one JSON document.

- A column that crosses a switch (the autoconversion threshold) has a
  kink: the derivative there is one-sided and does not match a central
  difference. 96 of 104 oracle columns pass `gradcheck`; the 8 that fail
  are threshold columns.
- Cost: see the plan's G1 paragraph. The adjoint is not within 3x of the
  kernel; hand-written adjoint kernels remain the way to meet it.
- `KesslerSeatModel` wraps the scheme as a column model for the NN seat
  ([nn-physics.md](nn-physics.md)).

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
