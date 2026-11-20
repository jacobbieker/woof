# Neural-network column physics (opt-in, inference only)

Phase P2 of [the differentiable/hybrid plan](dev/DIFFERENTIABLE-HYBRID-PLAN.md).
A **column model** maps the state of a model column to physical
tendencies. woof can run one in place of a physics scheme's tendencies, or
add one as a learned residual to them. A run that does either is **not
WRF-identical**. The identity door (`GPUWM_WRF_EXACT=1`) refuses it, and
every record of such a run says so.
A configuration that names no column model imports no NN runtime and
launches nothing extra; `tests/test_nn_default_path_imports.py` checks
this for the default doors.

This page covers the parts every seat shares: the artefact, the runners
and the safety layer. The seats, the configuration keys and the
training-data tap are described in the sections below as they land.

## The artefact

A model is a directory holding `manifest.json` and the weights file it
names (`woof/nn/manifest.py` documents every key):

| key | meaning |
|---|---|
| `schema` | `woof.nn.column-model/v1` |
| `name` | the model's name, carried into run records |
| `format` | `mlp-npz`, `mlx-safetensors`, `torchscript` or `torch-export` |
| `weights`, `weights_sha256` | the weights file and its SHA-256, checked on every load |
| `inputs`, `outputs` | `{name, units, levels}` per field, in feature order; level 0 is the lowest model level, and a surface field has `levels = 1` |
| `normalisation` | `input_mean`, `input_std`, `output_mean`, `output_std` per feature; the runners apply them |
| `bounds` | `input_min`, `input_max` per feature: the range the model was trained on |
| `output_limit` | optional `|output|` bound per feature |
| `architecture` | `{kind = "mlp", hidden = [...], activation}` for the MLP formats |
| `training` | free-form provenance |

`mlp-npz` (layers `w0, b0, w1, b1, ...`, each `w` shaped `(out, in)`)
runs on every backend, so a single artefact gives the same model on CUDA
and on Metal. `woof.nn.manifest.write_mlp_artefact` writes one. A
TorchScript or `torch.export` module takes `[column, n_inputs]`
normalised float32 and returns `[column, n_outputs]`; these formats run on
PyTorch: on CUDA, or on Metal with `backend = "torch"` (torch-MPS, see
"Runners and array exchange"). `mlx-safetensors` runs on Metal only.

The restart identity of a run binds the artefact's weights and manifest
hashes (`ColumnModelManifest.identity()`), so a resume onto different
weights is refused.

## Runners and array exchange

`woof.nn.models.load_column_model(path, backend="auto")` returns a
callable that maps `[n_inputs, column]` to `[n_outputs, column]` in
physical units, as arrays of the same kind as its input:

- **CUDA: PyTorch.** CuPy arrays reach torch through DLPack with no copy,
  and the result comes back the same way. The model runs on woof's current
  CuPy stream. torch is an optional extra, `recast-woof[nn-torch]`, and is
  never a base dependency. Its floor is the torch version the Earth2 door
  pins. The Earth2 doors run torch in a separate interpreter, which cannot
  share device memory with the forecast, so a column model needs torch in
  woof's own interpreter. When torch is missing, the refusal names the
  install line.
- **Metal: MLX.** The facade's arrays are views of `mx.array`s. The model
  reads that storage directly, and its result is wrapped as a new facade
  array, so the model joins woof's lazy graph without a host round trip.
  MLX is already part of `[gpu-mlx]`.
- **Metal: PyTorch (`backend = "torch"`).** torch-MPS reads the facade's
  buffer through DLPack with no copy, after the MLX queue drains, and the
  result comes back as one device copy (MLX cannot wrap a torch buffer).
  The tensor bridge, `woof.grad.bridge`, documents the ordering and the
  refusals ([grad.md](grad.md#the-tensor-bridge)).
- **Host: NumPy.** The reference runner for `mlp-npz`, used by the tests.

`backend = "auto"` follows woof's own backend. A format that the chosen
runtime cannot run is refused by name and is not converted.

## Safety

`woof.nn.safety.SafetyPolicy` wraps each model call:

- `out_of_range` decides what happens to columns with an input outside the
  trained `bounds`, widened by `margin` (a fraction of the range):
  - `fallback` (the default): those columns keep the physics scheme's
    tendency when the model replaces one, and get no NN increment when it
    adds a residual;
  - `refuse`: stop with the count and the feature most often out of range;
  - `clip`: clamp the inputs into the range.

  The counts are accumulated on the device and read once per report, so
  no per-step synchronisation is added.
- `limit_outputs` clamps each output to the manifest's `output_limit`.
- `positivity` keeps `q + dt * dq >= min(q, 0)` for moisture outputs. For
  a residual the floor applies to the sum of the scheme's rate and the
  model's.
- `fixer` (`none`, `moisture`, `moisture+heat`) shifts the tendency
  uniformly in mass within each column, so that its column integral equals
  that of the replaced scheme (or zero for a residual).

## Recording training data: the tendency tap (WRF route)

`woof run` and `woof go` take `--nn NN.toml`. That file holds one `[nn]`
table:

```toml
[nn]
mode = "tap"            # record only; "residual"/"replace" apply a model (below)
scheme = "pbl"          # pbl, radiation, cumulus or total
tap = "pbl-tap.zarr"    # relative to this file; *.icechunk makes an Icechunk repo
tap_every = 1           # every n-th dycore step
tap_columns = 256       # columns drawn per recorded step, interior only
tap_seed = 0
# levels = 30           # keep the lowest 30 levels of every 3-D field
# inputs = ["theta", "qv", "u", "v", "p", "z", "hfx", "ust"]
# outputs = ["dtheta", "dqv", "du", "dv"]
```

The seat sits where SPPT and SKEBS act: after the step's non-microphysics
physics and before chemistry, IAU and FDDA. On each recorded step it draws
`tap_columns` mass-grid columns at random from the interior. A specified or
nested domain's `spec_bdy_width` ring is excluded. The seat then writes the
following arrays:

- `inputs [sample, n_inputs, column]`: the named fields, levels bottom-up.
  3-D: `theta qv qc qr qi qs qg u v w p exner z dz rthraten`, with `u`, `v`
  and `w` averaged to mass points and `z` measured above the ground.
  Per column: `hgt mu hfx qfx ust tsk znt xland pblh psfc lh coszen`.
- `targets [sample, n_outputs, column]`: the scheme's held tendencies as
  physical rates (`dtheta dqv dqc dqr dqi dqs du dv`). They are decoupled
  by dividing by the coupler's response to a unit rate. On the interior
  this is the exact inverse of the dry-mass coupling in both numerics.
  `du` and `dv` are face-pair means, so they are approximate.
- `time_seconds`, `step`, `column_index` (`j * nx + i`).

The defaults depend on the scheme (`woof.nn.wrf_seat.DEFAULT_INPUTS`,
`DEFAULT_OUTPUTS`). The group attributes give the field specs in the
layout the artefact manifest takes, so a trainer can read
`woof.nn.tap.read_tap(path)` straight into `write_mlp_artefact`.

A scheme's slot holds the tendency of its last call, and `radt`, `cudt`
or a positive `bldt` hold it over steps. So a scheme is recorded only on
steps where it ran, and only on every `tap_every`-th dycore step the seat
sees. The record's `step` counts those steps: an adaptive clock changes
`dt` from step to step, so model time does not name a step, and a resumed
run starts counting again. The columns drawn depend only on `tap_seed` and
the model time, so a resumed run draws the same ones as an unbroken run.
A resumed run trims the store back to its resume time and appends from
there. `total` is recorded on
every such step, and it sums slots held from different steps.

A tap only reads. A tapped forecast is byte-identical to the same forecast
without `--nn`; `tests/test_nn_tap.py` checks this after real steps.
Recording is refused on a multi-domain experiment, on a `[tiles]`-streamed
domain, on an ensemble member leg, and on a store that already exists
unless the launch resumes (`--restart`, `--restart-from`).

## Applying a model (WRF route)

```toml
[nn]
mode = "residual"       # or "replace"
scheme = "pbl"          # pbl, radiation, cumulus or total
model = "pbl-mlp"       # the artefact directory, relative to this file
backend = "auto"        # MLX on Metal, PyTorch on CUDA; "torch", "mlx", "numpy"
out_of_range = "fallback"
margin = 0.0
fixer = "none"          # "moisture", "moisture+heat"
limit_outputs = true
positivity = true
# device = "cuda:0"
```

The inputs, outputs and levels are the artefact's. Every field must be in
the tap's vocabulary with the same units. The seat runs on every column at
every dycore step, from the time-t state, at the same place as the tap:

- `residual` adds the model's rate to the step's tendencies.
- `replace` uses the model's rate instead of the scheme's. The scheme
  still runs: its rate is what a fallback column keeps, and it sets the
  column budget the fixer reproduces. A scheme held over steps (`bldt`,
  `radt` or `cudt` > 0) is replaced on every step by a rate computed from
  that step's state.

The increment is coupled by `woof.core.physics.couple_column_tendencies`,
the couplers' own path, so the dry-mass weighting, the map factor and the
boundary-ring mask are the scheme's. It is added to a copy of the step's
tendencies, and the scheme's own slot is never written. Momentum goes from
the A-grid to the faces, so a replaced `du`/`dv` is the scheme's faces plus
the coupled difference of the two A-grid rates. Levels above the model's
keep the scheme.

The run prints one line at the end: the mode, the scheme, the artefact,
the runner, and how many column calls were outside the trained range.
`GPUWM_WRF_EXACT=1` refuses an applying mode at the door and again at
attach. A checkpoint written by an applying run carries the artefact's
hashes and the settings (`nn_column_seat` in the header). It resumes only
under the same `--nn` file and artefact, and a checkpoint written without a
model does not resume with one. The seat holds nothing between steps, so a
resumed run computes what the unbroken one does.
`tests/test_nn_apply.py` checks the arithmetic on the host, and the
residual, replace and zero-model runs on CUDA and Metal. A zero residual
leaves every prognostic value unchanged.

The physics registry lists both modes under `nn_column_physics` as
`experimental-runtime` with `wrf_identical: false`.

## Training a model, and a worked example (WRF route)

`python -m woof.nn.train TAP OUT` trains a small float32 MLP on a tap
store with NumPy alone, and writes an `mlp-npz` artefact that runs on every
backend. It uses Adam with minibatches. The last `--holdout` fraction of
recorded steps (default 0.2) is kept out, so the score is out of sample in
time. The manifest's statistics come from the training samples:
- normalisation;
- the trained input range that the safety screen uses;
- `output_limit` (twice the largest target seen per level, floored at a
  hundredth of the field's largest level limit, so a level that never saw a
  non-zero target is not clamped to zero online).

By default the tap's targets are the scheme's own tendency, so a trained
artefact is an emulator of the scheme, for `mode = "replace"`. A residual
model needs a reference target. On hex that target is the increment of
analysis nudging. Run with `--nudging CFG --nudging-analysis INIT ...` (grid
nudging toward analyses on the run's mesh, `woof.hex.nudging`) and tap with
`tap_target = "nudging"`. The record then holds, at each sampled step, the
nudging's increment to the held theta and moisture sources, decoupled into
rates exactly as the scheme's are: what the model lacked to stay on the
analyses. This follows Watt-Meyer et al.'s nudging-tendency correction. The
tap only reads: a nudged run that taps is byte-identical to the same
nudged run without `--nn`. `tap_target = "nudging"` is refused without
nudging, on the WRF route, and in an applying mode. The trainer marks each
artefact's `training.intended_mode` as `replace` (a physics tap) or
`residual` (a nudging tap), and both seats refuse an artefact in the
other mode. A sampled step the nudging leaves alone (past `gfdda_end_h`)
records nothing, so the record holds no false zero targets.
Samples are ordered by model time before the split, and a time recorded
twice (a resumed run appends) keeps its last record.

Several taps train one model: `python -m woof.nn.train TAP [TAP ...]
OUT` pools them (other cycles, seasons or meshes; their fields must
agree). Each store is split on its own, as above. A store named with
`--validate TAP` is held out whole, so the score is out of sample in
weather as well as in time; the `--holdout` tail then defaults to 0.
`--inputs a,b,...` keeps a subset of the tap's input fields and `--levels
N` the lowest N levels of every 3-D input and output, so one broad tap
serves several models. A store may be named only once, so a
`--validate` store is never also trained on. `--keep-best` keeps the
epoch with the lowest held-out error; the held-out scores are then those
of the selection (`selected_on_held_out` in the table), so score a further
cycle for an unbiased number. For each held-out store the report gives the R² per output
and, for each `--margins` value (default `0,0.1,0.3`), the fraction of its
columns outside the trained range widened by that margin: the share that
`out_of_range = "fallback"` would leave to the scheme. It also names the
input fields most often outside the range at the smallest margin, as the
share of columns with any level of the field outside.

Its `training` table records the taps, the splits, the hyper-parameters and
the held-out R² per output. The trainer demonstrates the workflow and is
not meant as a research trainer: anything that writes the same artefact
can be used in its place.

`tools/nn_seat_cost.py MODEL` measures what an applying seat adds to a
dycore step on this machine's backend. It also times the model call alone.

The worked example (P2.5) uses one real-data case: a 9 km, 80 x 100 x 49
domain run for 24 h with an adaptive time step (about 1400 steps), with
YSU, the revised MM5 surface layer and WSM6.

1. **Record.** `--nn` tap of `pbl` (default inputs and outputs, every 4th
   step, 256 columns). This gave 352 samples, and the tapped forecast's
   3849 history variables were byte-identical to the plain run's over all
   25 frames.
2. **Train.** `--hidden 256,256 --epochs 40` took 25 s on the CPU.
   Held-out R² (last 20 % of the day): dtheta 0.78, dqv 0.69, dqc 0.82,
   du 0.53, dv 0.57. (This artefact was trained before the per-level
   output-limit floor, so levels that saw no target were clamped to zero
   in the runs below.)
3. **Run online.** `mode = "replace"`, `scheme = "pbl"`, with fallback,
   output limits and positivity on, for the full 24 h. The table gives the
   RMSE against the physics run, averaged over the 25 hourly frames, on
   the interior (5-column ring excluded). TH10 is potential temperature
   over the lowest 10 levels. "Zero model" replaces the PBL tendency with
   zero, as a baseline for what dropping the scheme costs.

   | run | T2 (K) | U10 (m/s) | V10 (m/s) | Q2 (g/kg) | HFX (W/m²) | TH10 (K) |
   |---|---|---|---|---|---|---|
   | emulator, CUDA (torch) | 0.60 | 0.67 | 0.59 | 0.30 | 4.2 | 0.40 |
   | emulator, Metal (MLX) | 0.60 | 0.67 | 0.59 | 0.30 | 4.2 | 0.40 |
   | zero model, CUDA | 1.29 | 2.20 | 1.78 | 0.40 | 12.1 | 0.73 |

   The emulator run stayed stable for the whole day on both backends.
   About 14 % of column calls fell outside the trained range (one day of
   training data) and kept the scheme's tendency. The error grows from
   0.35 K T2 at 3 h to 1.0 K at 24 h.
4. **Restart.** A replace run resumed from its 12 h checkpoint reproduces
   the unbroken run byte for byte: every history variable over the last 12
   frames, and every final restart array.
5. **Cost.** `tools/nn_seat_cost.py`, same grid size (8000 columns, 349
   inputs, 245 outputs, 256 x 256 tanh MLP):

   | backend | dycore step | seat per step | model call |
   |---|---|---|---|
   | CUDA (A6000, shared with other jobs), torch | 29-30 ms | 5.3-6.0 ms | 3.3-3.7 ms |
   | CUDA, numpy runner (host round trip) | 31.5 ms | 265 ms | 125 ms |
   | Metal (MLX) | 32.8 ms | 3.4 ms | 1.05 ms |

   The model call is timed per call, each read back before the next (the
   latency a step pays). The CUDA rows are two runs on a shared card.

## The hex seat (MPAS route)

`woof hex forecast --nn NN.toml` (and `woof hex regional
--forecast-arg=--nn=/abs/NN.toml`) takes the same `[nn]` table on the MPAS
suite route, the hex default. The seat is in
`ResidentNativeCoupling.stage` (`woof/nn/hex_seat.py`): it acts on the
step's held physics sources once they are assembled from
`physics_get_tend`. That is after the regional padding and before SPPT/SKEB,
IAU and analysis nudging add theirs. So the tap records the schemes' own
sum, unperturbed. An applying model's rate is part of the physics source
that SPPT then perturbs, as it would perturb a scheme's. IAU and nudging
increments never reach the model.

Only `scheme = "total"` is available. The held source is the sum of every
non-microphysics scheme (PBL, surface layer, cumulus, radiation,
gravity-wave drag), and the lagged microphysics heating is held apart and
is not part of it. The source is held per unit dry mass, with `m = rho_zz`:

- `scalars[q] = m dq/dt`
- `rho_theta = m (c dtheta/dt + (Rv/Rd) theta dqv/dt)`, with
  `c = 1 + (Rv/Rd) qv` and `theta = theta_m / c` the dry potential
  temperature. This is MPAS's coupling into the modified potential
  temperature `theta_m`, so `rho_theta / m` is `dtheta_m/dt`.

The seat divides by `m` and inverts the theta conversion with the cell's
own `theta_m` and `qv`. An increment is coupled back the same way:

- `dtheta` is dry theta, in K s-1.
- `dtheta_m` is modified theta, in K s-1. A model writes one of the two.
- A `dq*` rate is in kg kg-1 s-1. A `dqv` increment also enters `theta_m`
  through the vapour term, as the scheme's does.

Momentum (`du`, `dv`) is refused, because the held momentum source is
edge-normal and the seat has no cell-to-edge projection yet.

The hex vocabulary has two kinds of input. The level-wise inputs, bottom-up,
are:

- `theta`, `theta_m`, `qv` ... `qg`;
- `u`, `v` (the reconstructed cell winds);
- `w` (the mean of the layer's two faces);
- `p`, `exner`, `rho` (dry density, `zz rho_zz`);
- `z` (layer centre above the ground), `dz`.

The cell inputs are `hgt`, `hfx`, `qfx`, `lh`, `ust`, `znt`, `pblh`, `tsk`,
`xland` and `psfc`, plus the radiation context `coszr` (cosine of the solar
zenith angle), `gsw` (net shortwave at the ground) and `glw` (downward
longwave at the ground), all as of the last radiation call (a run without
radiation leaves them at zero). They are read from the MPAS physics pools. `dx` is the cell's spacing in metres,
`sqrt(areaCell)` from the mesh (scaled by the Earth's radius on a
unit-sphere mesh): one model can then serve a variable-resolution mesh, or be
trained on taps pooled from meshes of different spacing.

On a limited-area mesh the columns follow the boundary rings:

- The tap draws only interior cells (`bdyMaskCell == 0`) and never the
  padded garbage cell. The draw is keyed by the seed and the model time, so
  a resumed run draws the same cells as the unbroken one.
- An applying seat writes every cell outside the specified zone
  (`bdyMaskCell < 6`, the cells nudging writes). The specified zone takes
  the boundary tendencies and is left alone.

What the seat refuses:

- an applying mode under `--physics-mode exact` or `GPUWM_WRF_EXACT=1`;
- `--members` rosters, `--samehost-devices`, `--hosts` and
  `--fullstep-local-timestep`, at the door;
- a moving region;
- the WOOF-physics route (`--physics ...`, `--physics-suite
  arwen_mpas_column`).

On the WOOF-physics route, a column model belongs in a backend row instead
(below).

An applying run seals the settings and the artefact's hashes in the
route's checkpoint carrier (`nn`). A resume checks them with
`woof.nn.hex_seat.check_carried`, and refuses a different model, a model
added to a checkpoint written without one, and the reverse. A native MPAS
restart file that carries no route carrier (an exact-mode restart) holds no
such identity, so the seat cannot check it. The model still screens the
specified zone's columns, which it never writes, so
`out_of_range = "refuse"` and the range counts see them too.

`tests/test_nn_hex_seat.py` checks on the host:

- that decoupling inverts MPAS's coupling;
- the tap's columns and rates;
- the residual, replace, positivity, fallback and refusal paths.

`tests/test_nn_hex_gpu.py` (opt-in, `WOOF_HEX_VARRES_CASE`) runs short
regional forecasts on a uniform and a variable-resolution mesh. It checks
three things:

- a tapped run is byte-identical to the plain run under fast and strict
  numerics, with its columns interior;
- a zero residual leaves every history variable unchanged;
- a constant `dtheta` residual warms the levels it spans by about
  rate x time.

### Results on hex (the plan's G2 gate)

Two regional meshes around one point, both on the MPAS suite route
(`convection_permitting`):

- uniform: 12 km, 250 km radius, 2107 cells, 55 levels;
- variable resolution: 3 km over a 100 km radius grading to 96 km, 4565
  cells.

Steps:

1. **Record.** A 120 h tap (`scheme = "total"`, 256 interior columns, every
   10th step on the uniform mesh and every 40th on the variable one) from a
   GFS cycle five days before the evaluation cycle. Each tapped run was
   byte-identical to its plain run (`tests/test_nn_hex_gpu.py` checks
   this on both meshes, fast and strict).
2. **Train.** `python -m woof.nn.train` with `--hidden 256,256 --epochs 20
   --holdout 0.1` took about 40 s per mesh on the CPU, on 138240 columns
   (540 recorded steps). The last 60 steps (10 %) were held out:

   | mesh | dtheta R² | dqv R² | dqc R² | (train R²) |
   |---|---|---|---|---|
   | uniform 12 km | 0.28 | 0.11 | 0.08 | 0.79-0.87 |
   | variable 3-96 km | -0.53 | -0.98 | -2.71 | 0.74-0.84 |

   Five days of one mesh do not generalise in time: the emulators are
   poor, and the variable-resolution one is worse than a zero model.
3. **Run online.** `mode = "replace"`, with fallback and `margin = 0.3`,
   120 h from the evaluation cycle. The margin was needed because pressure
   aloft in the evaluation week lies outside the training week's range.
   Even with it, most column calls fell back to the schemes:

   | mesh, backend | stable 120 h | column calls outside the range |
   |---|---|---|
   | uniform, CUDA (torch) | yes | 88 % (11.2 M of 12.6 M) |
   | uniform, Metal (MLX) | yes (max \|w\| 0.38 m/s) | 88 % |
   | variable, CUDA (torch) | yes | 89 % (97.4 M of 109.6 M) |

4. **Mass budget.** Area-mean surface pressure and precipitable water over
   the interior cells, compared with the physics run:

   | mesh | psfc change, physics / NN (Pa) | max \|NN - physics\| psfc | max \|NN - physics\| PW |
   |---|---|---|---|
   | uniform | -1742.9 / -1743.3 | 5.7 Pa | 0.37 mm |
   | variable | -1698.5 / -1699.4 | 14.1 Pa | 0.18 mm |

   The area-mean surface-pressure changes differ by under 1 Pa; the maxima
   are over all hourly frames. The NN runs keep the budget to the physics
   run's level.
5. **Scorecard.** The scorecard uses WIS2 land SYNOP T2 (lapse-corrected
   to the model height), 10 m wind and radiosonde temperature. Model and
   observation are paired by station and lead. The test is a station-block
   bootstrap (5000 draws) of the RMSE difference, NN minus physics; a
   difference above zero means the NN run is worse.

   | mesh | T2 RMSE physics / NN (K) | difference, 95 % CI | pairs, stations |
   |---|---|---|---|
   | uniform | 1.954 / 2.103 | +0.15 [+0.05, +0.26] (worse) | 8381, 141 |
   | variable | 1.511 / 1.450 | -0.06 [-0.11, -0.01] (better) | 5105, 86 |

   The 10 m wind and sonde scores have only one or two stations in these
   domains. They are too few to test.

6. **Cost.** The cost of the seat per step (2107 columns, 445 inputs, 165
   outputs, 256 x 256 tanh MLP) is the difference in the median step time
   between plain and NN runs of the same 6 h, run in pairs:

   | backend | plain step | NN step, minus plain, per pair |
   |---|---|---|
   | Metal (MLX) | 198-205 ms | +4.4, +1.5, +2.7 ms (1-2 %) |
   | CUDA (A6000 shared with other jobs), torch | 101-102 ms | +1.6, -29.6, -0.2, +10.7 ms |

   On CUDA the card was shared, so the step-time noise is larger than the
   seat's cost and the cost is not resolved there. The model call alone is 3.3-3.7 ms for 8000 columns on the
   WRF route (above).

The gate's verdict:

- **Pass:**
  - With the option absent, the identity tests are unchanged.
  - The NN runs are stable for 5 days on both meshes on CUDA, and on the
    uniform mesh on Metal.
  - The mass budget closes to the physics run's level.
  - Cost is measured on both backends: 1-2 % of a step on Metal; within
    the noise of a shared card on CUDA.
- **Fail:** the scorecard criterion, on the uniform mesh. T2 there is worse
  than the physics run at 95 %. The variable mesh is better at 95 %, but
  with 88-89 % of column calls falling back, the model acts in few
  columns. Neither result shows that the emulator has skill.

Passing the scorecard criterion needs broader training data (several
cycles and seasons, both meshes) so that fewer columns fall back, and a
better trainer than the NumPy example.

#### P2.6: broader data, a residual target, a dx input (G2 re-run)

The re-run follows the steps above with the same meshes, scorecard
(`tools/nn_g2_scorecard.py`) and tests. Every cycle is a 00 UTC GFS cycle in 2026. Training cycles,
validation cycle and evaluation cycle are disjoint, and the mode, model
and margin were chosen on the validation cycle alone:

- training: 02-10, 04-20, 07-15, 08-25, 10-12 and 11-02 (48 h uniform-mesh
  taps; 24 h variable-mesh taps of 02-10, 07-15 and 11-02);
- validation: 10-28 (120 h uniform, 72 h variable);
- evaluation: 11-07, 120 h. The WIS2 observations of that window end 59 h
  into the run, so the scored leads are 0-59 h.

The inputs add `dx` and the radiation context `coszr`, `gsw` and `glw`.

1. **Data breadth.** Offline fallback at `margin = 0.1` on the validation
   tap (uniform mesh, `replace`), against the number of training cycles:

   | cycles | columns outside the range | dtheta R² |
   |---|---|---|
   | 1 | 100 % | -0.81 |
   | 2 | 16 % | -0.51 |
   | 4 | 10 % | 0.20 |
   | 6 | 6.3 % | 0.31 (dqv 0.11, dqc 0.31) |

2. **Residual target.** `tap_target = "nudging"` (grid nudging toward
   6-hourly GFS analyses on the same mesh, 3e-4 s⁻¹) recorded the nudging
   increment on the same six cycles, and a `residual` model was trained on
   it (held-out dtheta R² 0.24, 0.25 against a zero model). Online on the
   validation cycle (uniform, 120 h, 15814 pairs at 132 stations, physics
   T2 RMSE 2.12 K):

   | arm | margin | T2 RMSE NN - physics (K), 95 % CI | calls falling back |
   |---|---|---|---|
   | replace | 0.1 | -0.069 [-0.111, -0.026] | 38 % |
   | replace | 0.3 | -0.066 [-0.113, -0.017] | 33 % |
   | replace | 1.0 | -0.062 [-0.110, -0.012] | 33 % |
   | residual (nudging increment) | 0.3 | +0.239 [+0.176, +0.305] | 21 % |
   | residual, 18 inputs | 0.3 | +0.172 [+0.120, +0.229] | 36 % |

   The residual arms cooled T2 by about 0.9 K. `replace` with margin 0.1
   was chosen.
3. **dx and the variable mesh.** A model pooled from the six uniform taps
   and the three variable-mesh taps covers the variable mesh's `dx`
   (14 % offline fallback on its validation tap, against 81 % for the
   uniform-only model, which falls back wherever `dx` is not 12 km). Online
   on the variable mesh's validation cycle (72 h, 5659 pairs at 78
   stations, physics 1.77 K) it was much worse, +1.09 [+0.97, +1.21] K
   (T2 bias -0.69 to -1.89 K). The uniform-only model was not worse,
   +0.03 [-0.04, +0.10] K, with 69 % of calls falling back. The
   uniform-only model was chosen for both meshes.
4. **Evaluation** (11-07, 120 h, `replace`, margin 0.1):

   | mesh, backend | stable 120 h | T2 RMSE physics / NN (K) | NN - physics, 95 % CI | pairs, stations | calls falling back |
   |---|---|---|---|---|---|
   | uniform, CUDA (torch) | yes | 2.056 / 2.000 | -0.056 [-0.094, -0.020] (better) | 7907, 133 | 94 % |
   | uniform, Metal (MLX) | yes | 2.005 / 1.979 | -0.026 [-0.069, +0.019] | 7907, 133 | 94 % |
   | variable, CUDA (torch) | yes | 1.621 / 1.579 | -0.042 [-0.096, +0.010] | 4628, 78 | 97 % |

   Mass budget (interior area means against the physics run, maxima over
   the hourly frames):

   | mesh, backend | psfc change, physics / NN (Pa) | max \|NN - physics\| psfc | max \|NN - physics\| PW |
   |---|---|---|---|
   | uniform, CUDA | -1728.4 / -1726.3 | 8.6 Pa | 0.39 mm |
   | uniform, Metal | -1728.1 / -1727.8 | 8.7 Pa | 0.39 mm |
   | variable, CUDA | -1690.2 / -1690.2 | 10.0 Pa | 0.14 mm |

   In the evaluation week nearly every column falls back. The cause is
   the top ten levels (about 40-10 hPa): early-November stratospheric
   theta and pressure lie below the range of all six training cycles.
   Below them, the columns lie inside it. On the validation
   week, 6 % of the tap's columns fell outside offline. Online, as the
   NN run's state drifted from the tap's, 33-38 % fell outside (69 % on the
   variable mesh). Nothing was chosen on the evaluation week.
5. **Cost** (plain and NN 6 h runs of the evaluation cycle in pairs,
   uniform mesh; median step):

   | backend | plain step | NN step, minus plain, per pair |
   |---|---|---|
   | Metal (MLX) | 188-197 ms | +0.6, +1.0, +3.6 ms (0.3-1.9 %) |
   | CUDA (A6000 shared with another job), torch | 91-93 ms | +7.3, +5.4, +8.0 ms (6-9 %) |

The gate's verdict: **met on its stated criteria.** With the option
absent, the identity tests are unchanged. The NN runs are stable for 5
days on both meshes (and on Metal on the uniform one). The mass budget
closes to the physics run's level, and cost is measured on both backends.
The T2 scorecard is not worse than the physics run at 95 % on either
mesh or backend, and is better at 95 % on the uniform mesh on CUDA.

The result is weak evidence of skill: 94-97 % of column calls fell back to
the schemes in the evaluation week. The next step is a model whose screen
covers the stratosphere: train only the lowest 44 levels (`--levels
44`; the schemes keep the levels above), or tap winter cycles. It needs a fresh evaluation
cycle, because this one has now been seen.

### A column model as a backend row

`woof.nn.hex_backend.NNColumnBackend` is an example
`woof.hex.physics_seam.PhysicsBackend`. It runs an inner backend on a
`ColumnBatch` and adds (`residual`) or substitutes (`replace`) a column
model's rates, under the same safety policy. It returns uncoupled rates,
which the seam's `unpack_mpas_tendencies` couples to dry mass. The batch's
winds are cell-centred, so `du`/`dv` are available there.
`tests/test_nn_hex_backend.py` runs it CUDA-free on
`DeterministicNumpyBackend`.

The example is not registered in woof's own metadata, because the row
lookup on every forecast would then import `woof.nn`. A third party ships
a row from its own distribution:

```toml
[project.entry-points."woof.hex.rows"]
my_nn_rows = "my_package.rows:rows"
```

Here `rows()` returns `PhysicsBackendRow` entries under new names (rows
are never rebound), and the row's adapter module publishes
`build_column_backend(row, **kw)`.

## A differentiable scheme through the seat

`woof.grad.kessler.KesslerSeatModel` is Kessler warm-rain microphysics as
a seat column model: it takes the seat's `[column, feature]` batch and
returns `dtheta, dqv, dqc, dqr` for one call of length `dt`. Because it is
torch, autograd gives its tangent-linear and adjoint through the seat's
vocabulary ([grad.md](grad.md#a-differentiable-scheme-kessler)). It is not
a trained model; it tests the seat's wiring and is not an accuracy claim.

## Installing

- CUDA: `pip install 'recast-woof[nn-torch]'`, with the CUDA build of
  torch from the index the Earth2 recipe names
  (`woof/authorities/earth2-models.v1.json`).
- Metal: nothing beyond `[gpu-mlx]`.

No weights ship in the wheel.
