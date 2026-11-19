"""woof.grad.bridge: the tensor bridge between woof arrays and PyTorch (plan P1.1).

Each exchange the bridge calls zero-copy is checked to share memory (a
pointer on CUDA and the host, a write through torch read back through MLX
on Metal), not merely to agree; each one it calls unsafe is checked to be
refused; and the ordering is checked by handing over a buffer the other
side is still writing.  Tests for a runtime this host lacks are skipped.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

torch = None


def _need_torch():
    global torch
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed (recast-woof[nn-torch])")
    import torch as _t
    torch = _t


def _backend():
    from woof import backend
    return backend.requested()


def _cupy_cuda():
    _need_torch()
    if _backend() != "cuda" or not torch.cuda.is_available():
        pytest.skip("needs CuPy on a CUDA card and a CUDA torch")
    import cupy
    if getattr(type(cupy.zeros(1)), "__woof_facade__", False):
        pytest.skip("cupy is the MLX facade here")
    return cupy


def _facade():
    _need_torch()
    if _backend() != "mlx" or not torch.backends.mps.is_available():
        pytest.skip("needs the Metal facade and torch-MPS")
    import cupy
    return cupy


# ---------------------------------------------------------------- host
def test_host_arrays_are_shared_on_cpu_and_refused_elsewhere():
    _need_torch()
    from woof.grad import bridge
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = bridge.to_torch(a)
    assert bridge.shares_memory(a, t)
    t[0, 0] = 7
    assert a[0, 0] == 7
    back = bridge.from_torch(t, a)
    assert np.shares_memory(back, a)
    with pytest.raises(bridge.BridgeRefused, match="copy=True"):
        bridge.to_torch(a, device="meta")
    with pytest.raises(bridge.BridgeRefused, match="non-contiguous"):
        bridge.to_torch(a[:, ::2])
    assert bridge.to_torch(a[:, ::2], copy=True).shape == (3, 2)
    with pytest.raises(bridge.BridgeRefused, match="exchanges"):
        bridge.to_torch(np.zeros(3, dtype=np.float16))


def test_autograd_never_gets_a_shared_view():
    _need_torch()
    from woof.grad import bridge
    a = np.linspace(1, 2, 5, dtype=np.float64)
    leaf = bridge.grad_leaf(a)
    assert leaf.requires_grad and leaf.is_leaf and not bridge.shares_memory(a, leaf)
    y = (leaf ** 2).sum()
    with pytest.raises(bridge.BridgeRefused, match="requires grad"):
        bridge.from_torch(leaf, a)
    y.backward()
    np.testing.assert_allclose(leaf.grad.numpy(), 2 * a)
    assert bridge.grad_leaf(a, dtype=np.float32).dtype == torch.float32


def test_a_woof_write_under_a_saved_view_fails_the_backward():
    _need_torch()
    from woof.grad import bridge
    a = np.linspace(1, 2, 5, dtype=np.float64)
    shared = bridge.to_torch(a)
    w = torch.ones(5, dtype=torch.float64, requires_grad=True)
    y = (w * shared).sum()          # autograd saves `shared` for dy/dw
    a[...] = 0.0                     # an in-place woof write torch cannot see
    bridge.mark_written(shared)
    with pytest.raises(RuntimeError, match="inplace"):
        y.backward()


# ---------------------------------------------------------------- CUDA
def test_cuda_exchange_is_zero_copy_both_ways():
    cupy = _cupy_cuda()
    from woof.grad import bridge
    x = cupy.arange(1000, dtype=cupy.float32).reshape(10, 100)
    t = bridge.to_torch(x)
    assert t.is_cuda and bridge.shares_memory(x, t)
    back = bridge.from_torch(t, x)
    assert back.data.ptr == x.data.ptr
    with pytest.raises(bridge.BridgeRefused, match="copy=True"):
        bridge.to_torch(x, device="cpu")
    assert bridge.to_torch(x, device="cpu", copy=True).device.type == "cpu"
    with pytest.raises(bridge.BridgeRefused, match="copy=True"):
        bridge.from_torch(torch.zeros(3), x)
    with pytest.raises(bridge.BridgeRefused, match="requires grad"):
        bridge.from_torch(torch.zeros(3, device="cuda", requires_grad=True), x)


def test_cuda_handoff_orders_the_two_streams():
    """woof writes on a busy non-blocking CuPy stream; torch reads on its
    own stream; then torch writes and woof reads on its stream."""
    cupy = _cupy_cuda()
    from woof.grad import bridge
    busy = cupy.ones((4096, 4096), cupy.float32)
    stream = cupy.cuda.Stream(non_blocking=True)
    for trial in range(3):
        with stream:
            for _ in range(8):
                busy = busy @ busy * cupy.float32(1.0 / 4096)
            x = cupy.full((1 << 20,), 3.0, cupy.float32) + busy[0, 0] * 0
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            with stream:
                t = bridge.to_torch(x)       # side waits for woof's stream
            got = float(t.sum().item())
        assert got == 3.0 * (1 << 20), trial
        with torch.cuda.stream(side):
            y = torch.full((1 << 20,), 5.0, device="cuda")
            for _ in range(8):
                y = y * 1.0
            with stream:
                back = bridge.from_torch(y, x)
                total = float(back.sum())
        assert total == 5.0 * (1 << 20), trial


def test_cuda_woof_stream_runs_torch_on_cupys_stream():
    cupy = _cupy_cuda()
    from woof.grad import bridge
    stream = cupy.cuda.Stream(non_blocking=True)
    x = cupy.ones(4, cupy.float32)
    with stream, bridge.woof_stream(x):
        assert torch.cuda.current_stream().cuda_stream == stream.ptr


# ---------------------------------------------------------------- Metal
def test_metal_export_is_zero_copy_and_ordered():
    cupy = _facade()
    from woof.grad import bridge
    x = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
    lazy = x * cupy.float32(2) + cupy.float32(1)   # still a pending MLX value
    t = bridge.to_torch(lazy)
    assert t.device.type == "mps"
    np.testing.assert_array_equal(t.cpu().numpy(), np.arange(6, dtype=np.float32).reshape(2, 3) * 2 + 1)
    assert bridge.shares_memory(lazy, t)
    t.mul_(10)
    bridge.release(lazy)
    np.testing.assert_array_equal(lazy.get(), (np.arange(6, dtype=np.float32).reshape(2, 3) * 2 + 1) * 10)
    # a view of a row shares the storage too
    row = x[1]
    tr = bridge.to_torch(row)
    tr.add_(100)
    bridge.release(row)
    np.testing.assert_array_equal(x.get()[1], [103, 104, 105])
    np.testing.assert_array_equal(x.get()[0], [0, 1, 2])


def test_metal_handback_is_one_copy_and_refusals():
    cupy = _facade()
    from woof.grad import bridge
    x = cupy.zeros((3,), dtype=cupy.float32)
    y = torch.arange(3, dtype=torch.float32, device="mps") * 2
    back = bridge.from_torch(y, x)
    assert getattr(type(back), "__woof_facade__", False)
    np.testing.assert_array_equal(back.get(), [0, 2, 4])
    y.add_(1)
    bridge.release(x)
    np.testing.assert_array_equal(back.get(), [0, 2, 4])    # a copy: later torch writes do not reach it
    with pytest.raises(bridge.BridgeRefused, match="float64"):
        bridge.to_torch(cupy.zeros(3, dtype=cupy.float64))
    with pytest.raises(bridge.BridgeRefused, match="copy=True"):
        bridge.to_torch(x, device="cpu")
    np.testing.assert_array_equal(bridge.to_torch(x, device="cpu", copy=True).numpy(), [0, 0, 0])
    with pytest.raises(bridge.BridgeRefused, match="copy=True"):
        bridge.from_torch(torch.zeros(3), x)


def test_torch_column_model_runs_on_metal_through_the_bridge(tmp_path):
    cupy = _facade()
    from woof.nn.manifest import FieldSpec, write_mlp_artefact
    from woof.nn.models import NumpyColumnModel, load_column_model
    rng = np.random.default_rng(0)
    layers = [(rng.normal(0, 0.4, (8, 4)), rng.normal(0, 0.1, 8)), (rng.normal(0, 0.4, (2, 8)), rng.normal(0, 0.1, 2))]
    m = write_mlp_artefact(
        tmp_path / "m", name="bridge-test", inputs=(FieldSpec("theta", "K", 4),),
        outputs=(FieldSpec("dtheta", "K s-1", 2),), layers=layers, activation="tanh",
        input_mean=np.full(4, 290.0), input_std=np.full(4, 5.0), output_mean=np.zeros(2),
        output_std=np.full(2, 1e-4), input_min=np.full(4, 270.0), input_max=np.full(4, 310.0))
    x = (290 + 5 * rng.standard_normal((4, 64))).astype(np.float32)
    model = load_column_model(m, backend="torch")
    assert str(model._device) == "mps"
    y = model(cupy.asarray(x))
    assert getattr(type(y), "__woof_facade__", False)
    np.testing.assert_allclose(y.get(), NumpyColumnModel(model.manifest)(x), rtol=1e-4, atol=1e-9)


def test_metal_float64_copies_keep_their_values():
    cupy = _facade()
    from woof.grad import bridge
    values = np.array([1.0 / 3.0, -2.5e-300, 6.02e23], dtype=np.float64)
    x = cupy.asarray(values)
    t = bridge.to_torch(x, device="cpu", copy=True)
    assert t.dtype == torch.float64
    np.testing.assert_array_equal(t.numpy(), values)
    back = bridge.from_torch(t * 2, x, copy=True)
    assert getattr(type(back), "__woof_facade__", False) and back.dtype == np.float64
    np.testing.assert_array_equal(back.get(), values * 2)


def test_torch_column_model_takes_a_float64_metal_input(tmp_path):
    cupy = _facade()
    from woof.nn.manifest import FieldSpec, write_mlp_artefact
    from woof.nn.models import NumpyColumnModel, load_column_model
    rng = np.random.default_rng(1)
    layers = [(rng.normal(0, 0.4, (2, 4)), rng.normal(0, 0.1, 2))]
    m = write_mlp_artefact(
        tmp_path / "m", name="bridge-f64", inputs=(FieldSpec("theta", "K", 4),),
        outputs=(FieldSpec("dtheta", "K s-1", 2),), layers=layers, activation="tanh",
        input_mean=np.full(4, 290.0), input_std=np.full(4, 5.0), output_mean=np.zeros(2),
        output_std=np.full(2, 1e-4), input_min=np.full(4, 270.0), input_max=np.full(4, 310.0))
    x = 290 + 5 * rng.standard_normal((4, 16))
    model = load_column_model(m, backend="torch")
    y = model(cupy.asarray(x))
    np.testing.assert_allclose(y.get(), NumpyColumnModel(model.manifest)(x.astype(np.float32)),
                               rtol=1e-4, atol=1e-9)
