# Neural-network column physics (opt-in, inference only)

Phase P2 of [the differentiable/hybrid plan](dev/DIFFERENTIABLE-HYBRID-PLAN.md).
A **column model** maps the state of a model column to physical
tendencies. woof can run one in place of a physics scheme's tendencies, or
add one as a learned residual to them. A run that does either is **not
WRF-identical**. The identity doors (`GPUWM_WRF_EXACT=1`,
`physics_mode = "exact"`) refuse it, and every record of such a run says so.
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
- `positivity` keeps `q + dt * dq >= 0` for moisture outputs.
- `fixer` (`none`, `moisture`, `moisture+heat`) shifts the tendency
  uniformly in mass within each column, so that its column integral equals
  that of the replaced scheme (or zero for a residual).

## Recording training data: the tendency tap (WRF route)

`woof run` and `woof go` take `--nn NN.toml`. That file holds one `[nn]`
table:

```toml
[nn]
mode = "tap"            # record only; applying a model comes later (P2.3)
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
steps where it ran and where the step is a multiple of `tap_every`. The
columns drawn depend only on `tap_seed` and the step, so a resumed run
draws the same ones as an unbroken run. A resumed run trims the store
back to its resume step and appends from there. `total` is recorded on
every such step, and it sums slots held from different steps.

A tap only reads. A tapped forecast is byte-identical to the same forecast
without `--nn`; `tests/test_nn_tap.py` checks this after real steps.
Recording is refused on a multi-domain experiment, on a `[tiles]`-streamed
domain, on an ensemble member leg, and on a store that already exists
unless the launch resumes (`--restart`, `--restart-from`).

## Installing

- CUDA: `pip install 'recast-woof[nn-torch]'`, with the CUDA build of
  torch from the index the Earth2 recipe names
  (`woof/authorities/earth2-models.v1.json`).
- Metal: nothing beyond `[gpu-mlx]`.

No weights ship in the wheel.
