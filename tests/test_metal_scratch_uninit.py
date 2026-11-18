"""woof.backend.scratch.scratch_empty: ``cp.empty`` without the Metal fill.

The facade's ``empty`` is ``zeros`` (a GPU pass over the buffer); scratch a
kernel writes before it reads (MYNN's worker arena) skips that pass.  The
words a kernel then writes and reads are the same: checked here for the
allocation itself, and for whole forecasts with WOOF_SCRATCH_POISON=1.
"""

import importlib.util

import numpy as np
import pytest


@pytest.fixture
def cp():
    pytest.importorskip("cupy")
    import cupy
    return cupy


def _reset():
    from woof.backend import scratch
    scratch._uninit_enabled.cache_clear()
    scratch._poison.cache_clear()


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize("shape", [0, 7, (3, 5), (2, 3, 4)])
def test_scratch_is_a_writable_c_contiguous_empty(cp, monkeypatch, dtype, shape):
    from woof.backend.scratch import scratch_empty
    _reset()
    try:
        a = scratch_empty(cp, shape, dtype)
        want = np.empty(shape, dtype)
        assert a.shape == want.shape and a.dtype == want.dtype and a.flags.c_contiguous
        host = np.arange(want.size).astype(dtype).reshape(want.shape)
        a[...] = cp.asarray(host)
        assert a.get().tobytes() == host.tobytes()
        monkeypatch.setenv("WOOF_SCRATCH_POISON", "1")
        _reset()
        b = scratch_empty(cp, shape, dtype)
        assert b.get().tobytes() == b"\xa5" * want.nbytes
    finally:
        monkeypatch.delenv("WOOF_SCRATCH_POISON", raising=False)
        _reset()


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs mlx (Apple silicon)")
def test_metal_scratch_skips_the_fill(cp, monkeypatch):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import mlx.core as mx
    from woof.backend.facade import _scratch
    from woof.backend.scratch import scratch_empty
    calls = []
    monkeypatch.setattr(mx, "zeros", lambda *a, **k: calls.append(a) or mx.full(*a, 0, **k))
    _reset()
    try:
        a = scratch_empty(cp, 1 << 20, cp.uint8)
        assert not calls and a.size == 1 << 20
        a[:8] = cp.asarray(np.arange(8, dtype=np.uint8))
        assert a[:8].get().tolist() == list(range(8))
        monkeypatch.setenv("WOOF_METAL_SCRATCH_UNINIT", "0")
        _reset()
        scratch_empty(cp, 16, cp.uint8)
        assert calls                          # the switch restores the filled empty
    finally:
        monkeypatch.delenv("WOOF_METAL_SCRATCH_UNINIT", raising=False)
        _reset()
    assert _scratch._kernel() is _scratch._kernel()
