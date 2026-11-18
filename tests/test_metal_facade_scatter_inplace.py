"""``a[prefix..., index] = v`` through the Metal facade, word for word.

An integer facade index on the last axis after a basic prefix (slices and
``...``) stores the selected elements in place with one kernel
(woof/backend/facade/_write.py ``scatter_last``) instead of scattering into a
copy of the whole array and writing every element back.  The reference is the
general path itself (``WOOF_METAL_SCATTER_INPLACE=0``) on a copy of the same
buffer, and NumPy: every word of the buffer must agree, NaN payloads and
negative indices included.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")


@pytest.fixture
def cp(monkeypatch):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


def _general(monkeypatch, fn):
    from woof.backend.facade import _write
    _write.scatter_enabled.cache_clear()
    monkeypatch.setenv("WOOF_METAL_SCATTER_INPLACE", "0")
    try:
        return fn()
    finally:
        monkeypatch.delenv("WOOF_METAL_SCATTER_INPLACE")
        _write.scatter_enabled.cache_clear()


def _host(rng, shape, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        words = rng.integers(0, 2**63, shape, dtype=np.uint64)
        bits = np.uint64 if dtype.itemsize == 8 else np.uint32
        return words.astype(bits).view(dtype).copy()
    return rng.integers(-2**30, 2**30, shape).astype(dtype)


_PREFIXES = [
    (Ellipsis,),
    (),
    (slice(None),),
    (slice(1, None), Ellipsis),
    (slice(None, None, 2), slice(1, -1)),
    (slice(None, None, -1), Ellipsis),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint32])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_scatter_matches_general_path_and_numpy(cp, monkeypatch, dtype, index_dtype):
    from woof.backend.facade import _write
    rng = np.random.default_rng(11)
    calls = []
    real = _write.scatter_last

    def counting(*a, **k):
        done = real(*a, **k)
        calls.append(done)
        return done
    monkeypatch.setattr(_write, "scatter_last", counting)
    for shape in [(37,), (3, 29), (4, 5, 23), (2, 3, 4, 17)]:
        for prefix in _PREFIXES:
            n_basic = sum(1 for k in prefix if k is not Ellipsis)
            if Ellipsis not in prefix and n_basic != len(shape) - 1:
                continue
            if n_basic > len(shape) - 1:
                continue
            n = shape[-1]
            m = int(rng.integers(1, n))
            index = rng.choice(np.arange(-n, n), m, replace=False)
            index = index[np.unique(index % n, return_index=True)[1]]     # no repeats
            host = _host(rng, shape, dtype)
            key_host = prefix + (index.astype(index_dtype),)
            sel = host[key_host]
            for kind in ("array", "scalar", "row"):
                if kind == "array":
                    value_host = _host(rng, sel.shape, dtype)
                elif kind == "scalar":
                    value_host = _host(rng, (), dtype)[()]
                else:
                    value_host = _host(rng, sel.shape[-1:], dtype)
                want = host.copy()
                want[key_host] = value_host
                key = prefix + (cp.asarray(index.astype(index_dtype)),)
                value = cp.asarray(value_host) if kind != "scalar" else value_host
                fast = cp.asarray(host)
                fast[key] = value
                got = fast.get()
                assert got.tobytes() == want.tobytes(), (shape, prefix, kind)
                general = cp.asarray(host)
                if np.dtype(dtype) == np.int64:
                    # MLX's GPU scatter has no int64: the general path refuses
                    with pytest.raises(ValueError, match="int64"):
                        _general(monkeypatch, lambda: general.__setitem__(key, value))
                    continue
                _general(monkeypatch, lambda: general.__setitem__(key, value))
                assert got.tobytes() == general.get().tobytes(), (shape, prefix, kind)
    assert calls and calls.count(True) * 2 == len(calls)      # every fast store, no general one


def test_out_of_place_keys_take_the_general_path(cp):
    from woof.backend.facade._write import scatter_last
    a = cp.arange(24, dtype=cp.float32).reshape(4, 6)
    idx = cp.asarray(np.array([0, 2], dtype=np.int64))
    # an index on a leading axis, an integer prefix, a host index, a boolean
    # mask, a 2-D index and a value sharing the buffer all decline
    assert not scatter_last(a, (idx, slice(None)), 1.0)
    assert not scatter_last(a, (1, idx), 1.0)
    assert not scatter_last(a, (Ellipsis, np.array([0, 2])), 1.0)
    assert not scatter_last(a, (Ellipsis, cp.asarray(np.array([True] * 6))), 1.0)
    assert not scatter_last(a, (Ellipsis, cp.asarray(np.array([[0], [2]]))), 1.0)
    assert not scatter_last(a, (Ellipsis, idx), a[:, :2])
    host = np.arange(24, dtype=np.float32).reshape(4, 6)
    a[idx, :] = -1.0
    a[1, idx] = -2.0
    a[..., idx] = a[:, 3:5]
    host[[0, 2], :] = -1.0
    host[1, [0, 2]] = -2.0
    host[..., [0, 2]] = host[:, 3:5]
    assert a.get().tobytes() == host.tobytes()


def test_scatter_orders_after_readers_and_before_later_launches(cp):
    """The store is an in-place launch: a lazy reader of the old values sees
    them, and a later reader sees the new ones."""
    a = cp.asarray(np.arange(40, dtype=np.float32).reshape(4, 10))
    before = a * 1.0                         # lazy reader of the old words
    idx = cp.asarray(np.array([1, 3, -1], dtype=np.int32))
    a[..., idx] = cp.asarray(np.full((4, 3), 7.5, dtype=np.float32))
    after = a + 0.0
    host = np.arange(40, dtype=np.float32).reshape(4, 10)
    assert before.get().tobytes() == host.tobytes()
    host[..., [1, 3, -1]] = 7.5
    assert after.get().tobytes() == host.tobytes()
    assert a.get().tobytes() == host.tobytes()


def test_empty_selection_and_deep_outer_axes_decline(cp):
    from woof.backend.facade._write import scatter_last
    a = cp.zeros((4, 6), dtype=cp.float32)
    empty = cp.asarray(np.array([], dtype=np.int32))
    # nothing to store: the general path keeps checking the value's shape
    assert not scatter_last(a, (Ellipsis, empty), cp.ones((7, 9), dtype=cp.float32))
    # nine outer axes that do not collapse (every other element) exceed the kernel's eight
    deep = cp.zeros((3,) * 9 + (4,), dtype=cp.float32)[(slice(None, None, 2),) * 9]
    assert not scatter_last(deep, (Ellipsis, cp.asarray(np.array([1], dtype=np.int32))), 1.0)


@pytest.mark.parametrize("key", ["leading", "ellipsis", "slices", "short"])
def test_eight_byte_general_path_keys(cp, monkeypatch, key):
    """The general path stores 8-byte words through a uint32 halves view: every
    key form must leave the halves axis alone."""
    rng = np.random.default_rng(5)
    host = rng.standard_normal((3, 4, 5))
    idx = np.array([0, 2], dtype=np.int32)
    key_host = {"leading": (idx,), "ellipsis": (Ellipsis, idx),
                "slices": (slice(None), slice(None), idx), "short": (slice(None), idx)}[key]
    want = host.copy()
    want[key_host] = -3.25
    a = cp.asarray(host)
    dkey = tuple(cp.asarray(k) if isinstance(k, np.ndarray) else k for k in key_host)
    _general(monkeypatch, lambda: a.__setitem__(dkey, -3.25))
    assert a.get().tobytes() == want.tobytes()
