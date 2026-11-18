"""woof.backend.scratch.scratch_empty: ``cp.empty`` without the Metal fill.

The facade's ``empty`` is ``zeros`` (a GPU pass over the buffer); scratch a
kernel writes before it reads (MYNN's worker arena) skips that pass.  This
file checks the allocation; that every read word is written first is shown
by whole forecasts poisoned with two bytes (WOOF_SCRATCH_POISON=ff and a5),
word-identical to the ordinary run on CUDA and Metal (the PR's evidence).
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
        for knob, byte in (("1", b"\xff"), ("a5", b"\xa5")):
            monkeypatch.setenv("WOOF_SCRATCH_POISON", knob)
            _reset()
            b = scratch_empty(cp, shape, dtype)
            assert b.get().tobytes() == byte * want.nbytes
    finally:
        monkeypatch.delenv("WOOF_SCRATCH_POISON", raising=False)
        _reset()


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs mlx (Apple silicon)")
def test_metal_scratch_skips_the_fill(cp, monkeypatch):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import mlx.core as mx
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


@pytest.mark.parametrize("value, want", [("0", None), ("off", None), ("false", None), ("no", None),
                                         ("1", 0xFF), ("on", 0xFF), ("a5", 0xA5), ("00", 0)])
def test_switch_values(monkeypatch, value, want):
    from woof.backend import scratch
    monkeypatch.setenv("WOOF_SCRATCH_POISON", value)
    _reset()
    try:
        assert scratch._poison() == want
    finally:
        monkeypatch.delenv("WOOF_SCRATCH_POISON")
        _reset()
    for off in ("0", "off", "False", "no"):
        monkeypatch.setenv("WOOF_METAL_SCRATCH_UNINIT", off)
        _reset()
        assert not scratch._uninit_enabled()
    monkeypatch.delenv("WOOF_METAL_SCRATCH_UNINIT")
    _reset()
    assert scratch._uninit_enabled()


@pytest.mark.parametrize("order", ["C", "F"])
def test_scratch_like_keeps_the_layout_of_its_input(cp, monkeypatch, order):
    """``scratch_like`` replaces ``empty_like``: an F-ordered input keeps an
    F-ordered output, so a kernel writing it linearly maps the same elements."""
    from woof.backend.scratch import scratch_like
    a = cp.asarray(np.asfortranarray(np.zeros((3, 5), np.float32)) if order == "F"
                   else np.zeros((3, 5), np.float32))
    if order == "F" and a.flags.c_contiguous:
        pytest.skip("this array module holds F-ordered arrays C-contiguous")
    _reset()
    try:
        out = scratch_like(cp, a)
        assert out.shape == a.shape and out.dtype == a.dtype
        assert out.strides == cp.empty_like(a).strides
        monkeypatch.setenv("WOOF_SCRATCH_POISON", "a5")
        _reset()
        assert scratch_like(cp, a).get().tobytes() == b"\xa5" * a.nbytes
    finally:
        monkeypatch.delenv("WOOF_SCRATCH_POISON", raising=False)
        _reset()
