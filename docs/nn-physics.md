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
CUDA only. `mlx-safetensors` runs on Metal only.

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
`xland` and `psfc`. They are read from the MPAS physics pools.

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

## Installing

- CUDA: `pip install 'recast-woof[nn-torch]'`, with the CUDA build of
  torch from the index the Earth2 recipe names
  (`woof/authorities/earth2-models.v1.json`).
- Metal: nothing beyond `[gpu-mlx]`.

No weights ship in the wheel.
