"""woof.nn: the column-model artefact, its runners and the safety layer.

One ``mlp-npz`` artefact is run by every runner this host has (NumPy
always; MLX on Apple silicon, through the facade when it is the active
backend; PyTorch where installed, on CUDA when there is a card) and each
result is held to the NumPy reference.  On CUDA the exchange is checked to
share memory (DLPack), not merely to agree.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from woof.nn import manifest as nn_manifest
from woof.nn.manifest import FieldSpec, ManifestError, load_manifest, write_mlp_artefact
from woof.nn.models import NumpyColumnModel, load_column_model
from woof.nn import safety

LEVELS = 6


def _artefact(tmp_path, *, activation="tanh", hidden=(16, 8), seed=0):
    rng = np.random.default_rng(seed)
    inputs = (FieldSpec("theta", "K", LEVELS), FieldSpec("qv", "kg kg-1", LEVELS), FieldSpec("hfx", "W m-2", 1))
    outputs = (FieldSpec("dtheta", "K s-1", LEVELS), FieldSpec("dqv", "kg kg-1 s-1", LEVELS))
    n_in, n_out = 2 * LEVELS + 1, 2 * LEVELS
    widths = [n_in, *hidden, n_out]
    layers = [(rng.normal(0, 0.4, (widths[i + 1], widths[i])), rng.normal(0, 0.1, widths[i + 1]))
              for i in range(len(widths) - 1)]
    in_mean = np.r_[np.full(LEVELS, 290.0), np.full(LEVELS, 8e-3), 50.0]
    in_std = np.r_[np.full(LEVELS, 5.0), np.full(LEVELS, 2e-3), 40.0]
    return write_mlp_artefact(
        tmp_path / "model", name="test-emulator", inputs=inputs, outputs=outputs, layers=layers,
        activation=activation, input_mean=in_mean, input_std=in_std,
        output_mean=np.zeros(n_out), output_std=np.r_[np.full(LEVELS, 1e-4), np.full(LEVELS, 1e-7)],
        input_min=in_mean - 3 * in_std, input_max=in_mean + 3 * in_std,
        output_limit=np.r_[np.full(LEVELS, 1e-3), np.full(LEVELS, 1e-6)],
        training={"source": "synthetic"})


def _inputs(manifest, columns=257, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (manifest.n_inputs, columns)).astype(np.float32)
    return manifest.input_mean[:, None] + z * manifest.input_std[:, None]


def _reference(manifest, x):
    act = {"tanh": np.tanh, "relu": lambda h: np.maximum(h, 0)}[manifest.architecture["activation"]]
    h = (x.astype(np.float64) - manifest.input_mean[:, None]) / manifest.input_std[:, None]
    layers = nn_manifest.mlp_layers(manifest)
    for i, (w, b) in enumerate(layers):
        h = w.astype(np.float64) @ h + b[:, None]
        if i < len(layers) - 1:
            h = act(h)
    return h * manifest.output_std[:, None] + manifest.output_mean[:, None]


def _close(a, b, manifest):
    scale = manifest.output_std[:, None]
    np.testing.assert_allclose(np.asarray(a) / scale, np.asarray(b) / scale, rtol=0, atol=2e-5)


# ------------------------------------------------------------------ manifest
def test_manifest_round_trip_and_identity(tmp_path):
    m = _artefact(tmp_path)
    assert m.n_inputs == 2 * LEVELS + 1 and m.n_outputs == 2 * LEVELS
    assert m.input_slices()["hfx"] == slice(2 * LEVELS, 2 * LEVELS + 1)
    ident = m.identity()
    assert ident["schema"] == nn_manifest.SCHEMA and len(ident["weights_sha256"]) == 64
    assert load_manifest(m.path).identity() == ident
    assert load_manifest(m.path.parent).identity() == ident


def test_manifest_refuses_changed_weights(tmp_path):
    m = _artefact(tmp_path)
    data = dict(np.load(m.weights))
    data["b0"] = data["b0"] + 1
    np.savez(m.weights, **data)
    with pytest.raises(ManifestError, match="SHA-256"):
        load_manifest(m.path)


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d.update(format="onnx"), "format"),
    (lambda d: d.update(extra=1), "unknown key"),
    (lambda d: d["normalisation"].update(input_std=[0.0] * len(d["normalisation"]["input_std"])), "positive"),
    (lambda d: d["bounds"].update(input_min=d["bounds"]["input_min"][:-1]), "entries"),
    (lambda d: d["bounds"].update(input_min=[1e9] * len(d["bounds"]["input_min"])), "exceeds"),
    (lambda d: d["architecture"].update(activation="softsign"), "activation"),
    (lambda d: d["inputs"].append(dict(d["inputs"][0])), "repeats"),
])
def test_manifest_refusals(tmp_path, edit, message):
    m = _artefact(tmp_path)
    raw = json.loads(m.path.read_text())
    edit(raw)
    m.path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match=message):
        load_manifest(m.path)


def test_mlp_layers_refuse_wrong_shapes(tmp_path):
    m = _artefact(tmp_path)
    data = dict(np.load(m.weights))
    data["w1"] = data["w1"][:, :-1]
    np.savez(m.weights, **data)
    raw = json.loads(m.path.read_text())
    raw["weights_sha256"] = nn_manifest._sha256(m.weights)
    m.path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match="layer 1"):
        nn_manifest.mlp_layers(load_manifest(m.path))


# ------------------------------------------------------------------ runners
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_numpy_runner_matches_float64_reference(tmp_path, activation):
    m = _artefact(tmp_path, activation=activation)
    x = _inputs(m)
    y = NumpyColumnModel(m)(x)
    assert y.dtype == np.float32 and y.shape == (m.n_outputs, x.shape[1])
    _close(y, _reference(m, x), m)


def test_load_column_model_refuses_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="backend"):
        load_column_model(_artefact(tmp_path), backend="onnx")


def test_mlx_runner_matches_reference(tmp_path):
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX is not installed (Apple silicon only)")
    m = _artefact(tmp_path)
    x = _inputs(m)
    model = load_column_model(m, backend="mlx")
    y = model(x)
    assert isinstance(y, np.ndarray)
    _close(y, _reference(m, x), m)


def test_mlx_runner_takes_and_returns_facade_arrays(tmp_path):
    from woof.backend import requested
    if requested() != "mlx":
        pytest.skip("the MLX facade is not this process's backend")
    import cupy  # the facade
    m = _artefact(tmp_path)
    x = _inputs(m)
    device_x = cupy.asarray(x)
    y = load_column_model(m, backend="mlx")(device_x)
    assert getattr(type(y), "__woof_facade__", False) and y.shape == (m.n_outputs, x.shape[1])
    _close(y.get(), _reference(m, x), m)
    # the input's mx.array is the facade's own storage view, not a host copy
    from woof.nn.bridge import to_mlx
    assert to_mlx(device_x).shape == device_x.shape


def test_torch_runner_matches_reference_on_host(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed (recast-woof[nn-torch])")
    m = _artefact(tmp_path)
    x = _inputs(m)
    y = load_column_model(m, backend="torch", device="cpu")(x)
    assert isinstance(y, np.ndarray)
    _close(y, _reference(m, x), m)


def test_torch_runner_shares_cupy_memory_on_cuda(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed (recast-woof[nn-torch])")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device for PyTorch")
    import cupy
    if getattr(type(cupy.zeros(1)), "__woof_facade__", False):
        pytest.skip("cupy is the MLX facade here")
    from woof.nn.bridge import from_torch, to_torch
    m = _artefact(tmp_path)
    x = cupy.asarray(_inputs(m))
    t = to_torch(x)
    assert t.data_ptr() == x.data.ptr and t.is_cuda
    back = from_torch(t, x)
    assert back.data.ptr == x.data.ptr
    model = load_column_model(m, backend="torch")
    y = model(x)
    assert isinstance(y, cupy.ndarray) and y.shape == (m.n_outputs, x.shape[1])
    _close(y.get(), _reference(m, x.get()), m)


def test_runners_refuse_formats_they_cannot_run(tmp_path):
    m = _artefact(tmp_path)
    from dataclasses import replace
    from woof.nn.models import ModelUnavailable
    with pytest.raises(ModelUnavailable):
        NumpyColumnModel(replace(m, format="torchscript"))
    if importlib.util.find_spec("mlx") is not None:
        from woof.nn.models import MlxColumnModel
        with pytest.raises(ModelUnavailable, match="CUDA"):
            MlxColumnModel(replace(m, format="torch-export"))


# ------------------------------------------------------------------ safety
def test_out_of_range_policies(tmp_path):
    m = _artefact(tmp_path)
    x = _inputs(m, columns=8)
    x[0, 3] = m.input_max[0] + 1.0
    x[LEVELS, 5] = np.nan
    mask = safety.out_of_range_columns(x, m.input_min, m.input_max)
    assert mask.tolist() == [False, False, False, True, False, True, False, False]
    policy = safety.SafetyPolicy(out_of_range="fallback")
    _, fallback = policy.screen(x, m)
    assert fallback.tolist() == mask.tolist()
    assert policy.ledger.summary()["columns_out_of_range"] == 2
    with pytest.raises(safety.InputOutOfRange, match="2 of 8 columns"):
        safety.SafetyPolicy(out_of_range="refuse").screen(x, m, feature_names=["a"] * m.n_inputs)
    clipped, finite_fallback = safety.SafetyPolicy(out_of_range="clip").screen(x[:, :5], m)
    assert not finite_fallback.any() and clipped[0, 3] == pytest.approx(m.input_max[0])
    # a margin widens the range by a fraction of it
    wide = safety.out_of_range_columns(x[:, :5], m.input_min, m.input_max, margin=0.2)
    assert not wide.any()


def test_limits_positivity_and_column_fix():
    y = np.array([[3.0, -3.0], [0.5, -0.5]], dtype=np.float32)
    np.testing.assert_array_equal(safety.limit_outputs(y, [1.0, 1.0]), [[1, -1], [0.5, -0.5]])
    q = np.array([[1e-3, 0.0], [2e-3, 5e-4]], dtype=np.float32)
    dq = np.array([[-1.0, -1.0], [1e-5, -1e-3]], dtype=np.float32)
    out = safety.positive_tendency(dq, q, dt=10.0)
    assert np.all(q + 10.0 * out >= -1e-9)
    weight = np.array([[100.0, 0.0], [300.0, 0.0]], dtype=np.float32)
    d = np.array([[1.0, 2.0], [-1.0, 3.0]], dtype=np.float32)
    fixed = safety.column_fix(d, weight, np.array([0.0, 7.0], dtype=np.float32))
    assert float((weight[:, 0] * fixed[:, 0]).sum()) == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_array_equal(fixed[:, 1], d[:, 1])   # a massless column is left alone
    with pytest.raises(ValueError):
        safety.SafetyPolicy(fixer="energy")
    with pytest.raises(ValueError):
        safety.SafetyPolicy(out_of_range="ignore")


# ------------------------------------------------------------------ review follow-ups
def test_clip_policy_still_falls_back_on_non_finite_columns(tmp_path):
    m = _artefact(tmp_path)
    x = _inputs(m, columns=4)
    x[0, 1] = m.input_max[0] + 10.0
    x[1, 2] = np.inf
    clipped, fallback = safety.SafetyPolicy(out_of_range="clip").screen(x, m)
    assert fallback.tolist() == [False, False, True, False]
    assert clipped[0, 1] == pytest.approx(m.input_max[0])


def test_constrain_applies_limits_fixer_and_positivity(tmp_path):
    m = _artefact(tmp_path)
    columns = 5
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.normal(0, 5e-4, (LEVELS, columns)),
                        rng.normal(0, 2e-6, (LEVELS, columns))]).astype(np.float32)
    qv = np.full((LEVELS, columns), 1e-5, dtype=np.float32)
    weight = np.full((LEVELS, columns), 100.0, dtype=np.float32)
    exner = np.full((LEVELS, columns), 0.9, dtype=np.float32)
    target = np.full(columns, 1e-4, dtype=np.float32)
    policy = safety.SafetyPolicy(fixer="moisture+heat")
    out = policy.constrain(y, m, dt=60.0, state={"qv": qv, "exner": exner}, weight=weight,
                           targets={"dqv": target})
    rows = m.output_slices()
    dtheta, dqv = out[rows["dtheta"]], out[rows["dqv"]]
    np.testing.assert_allclose((weight * exner * dtheta).sum(axis=0), 0.0, atol=1e-6)
    assert np.all(qv + 60.0 * dqv >= -1e-12)
    # untouched rules give the limited array back
    plain = safety.SafetyPolicy(positivity=False, limit_outputs=False).constrain(y, m, dt=60.0)
    assert plain is y
    with pytest.raises(ValueError, match="exner"):
        policy.constrain(y, m, dt=60.0, state={"qv": qv}, weight=weight)


def test_torch_runner_casts_float64_input(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed (recast-woof[nn-torch])")
    m = _artefact(tmp_path)
    x = _inputs(m).astype(np.float64)
    y = load_column_model(m, backend="torch", device="cpu")(x)
    assert y.dtype == np.float32
    _close(y, _reference(m, x), m)


def test_mlx_safetensors_reads_sequential_layer_numbering(tmp_path):
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX is not installed (Apple silicon only)")
    import mlx.core as mx
    import mlx.nn as nn
    from woof.nn.models import MlxColumnModel
    m = _artefact(tmp_path)
    layers = nn_manifest.mlp_layers(m)
    seq = nn.Sequential(nn.Linear(1, 1), nn.Tanh(), nn.Linear(1, 1), nn.Tanh(), nn.Linear(1, 1))
    for module, (w, b) in zip([seq.layers[0], seq.layers[2], seq.layers[4]], layers):
        module.weight, module.bias = mx.array(w), mx.array(b)
    target = m.path.parent / "weights.safetensors"
    seq.save_weights(str(target))
    raw = json.loads(m.path.read_text())
    raw.update(format="mlx-safetensors", weights=target.name, weights_sha256=nn_manifest._sha256(target))
    m.path.write_text(json.dumps(raw))
    st = load_manifest(m.path)
    x = _inputs(st)
    _close(MlxColumnModel(st)(x), _reference(m, x), m)
    # a transposed layer is refused at load, not discovered inside addmm
    seq.layers[2].weight = mx.array(layers[1][0].T.copy())
    seq.save_weights(str(target))
    raw["weights_sha256"] = nn_manifest._sha256(target)
    m.path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match="layer 2"):
        MlxColumnModel(load_manifest(m.path))


def test_bridge_kinds_include_the_host_facade():
    from woof.backend import hostfacade
    from woof.nn import bridge
    a = hostfacade.asarray(np.ones((2, 3), dtype=np.float32))
    assert bridge.array_kind(a) == "host"
    assert bridge.array_kind(np.ones(2)) == "numpy"
    assert bridge.host_view(a).shape == (2, 3)
