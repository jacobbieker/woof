"""Fast-numerics ``a op= b`` through the Metal facade, word for word.

``a op= b`` on a binary32 view runs one kernel (woof/backend/facade/_write.py
``arith_in_place``) instead of MLX's op into a new array plus a strided copy.
The reference is that general path itself, run on a copy of the same buffer:
every word of the buffer must agree, subnormal and non-finite operands
included.  An operand sharing the target's buffer keeps the general path.
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
    monkeypatch.setenv("GPUWM_WRF_EXACT", "0")
    import cupy
    return cupy


def test_pair_walk_matches_every_view():
    from woof.backend.facade._write import _collapsed_pair
    rng = np.random.default_rng(5)
    for _ in range(2000):
        nd = int(rng.integers(1, 5))
        shape = tuple(int(x) for x in rng.integers(1, 6, nd))
        dst = [int(x) for x in rng.integers(-40, 40, nd)]
        src = [int(x) if rng.random() < 0.7 else 0 for x in rng.integers(-40, 40, nd)]
        if rng.random() < 0.5:      # a dense row-major walk somewhere
            step = 1
            for d in range(nd - 1, -1, -1):
                dst[d] = step
                step *= shape[d]
        n, a, b = _collapsed_pair(shape, dst, src)
        got = []
        for i in range(int(np.prod(shape))):
            rem, oa, ob = i, 0, 0
            for d in range(len(n) - 1, 0, -1):
                k = rem % n[d]
                oa += k * a[d]
                ob += k * b[d]
                rem //= n[d]
            got.append((oa + rem * a[0], ob + rem * b[0]))
        want = [(sum(i * s for i, s in zip(idx, dst)), sum(i * s for i, s in zip(idx, src)))
                for idx in np.ndindex(shape)]
        assert got == want


def _words(rng, shape):
    w = rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    # a third of the words subnormal or tiny, a few non-finite
    small = rng.random(shape) < 0.33
    w[small] = (w[small] & np.uint32(0x80FFFFFF)) | (rng.integers(0, 3, int(small.sum())).astype(np.uint32) << 23)
    v = w.view(np.float32)
    v[rng.random(shape) < 0.01] = np.inf
    v[rng.random(shape) < 0.01] = np.nan
    return v


_KEYS = [
    (Ellipsis,),
    (slice(None), slice(2, -2), slice(2, -2)),
    (slice(None), slice(-2, None), slice(None)),
    (slice(1, None, 2), slice(None, None, -1), slice(None)),
    (2, slice(None), slice(4, 9)),
]


def _operands(cp, rng, shape):
    yield "scalar", 0.1
    yield "int", 3
    yield "np.float32", np.float32(1e-39)
    yield "same", cp.asarray(_words(rng, shape))
    whole = cp.asarray(_words(rng, (shape[0] + 1, shape[1] + 3, shape[2] + 2)))
    yield "strided", whole[1:, 1:-2, ::-1][..., :shape[2]]
    yield "row", cp.asarray(_words(rng, shape[1:]))[None]
    yield "column", cp.asarray(_words(rng, (shape[0], 1, 1)))


@pytest.mark.parametrize("op", ["__iadd__", "__isub__", "__imul__", "__itruediv__"])
@pytest.mark.parametrize("key", _KEYS, ids=[str(i) for i in range(len(_KEYS))])
def test_fused_in_place_matches_general_path(cp, op, key):
    from woof.backend.facade import _write
    from woof.backend.facade._ops import operand_values, UFUNCS
    name = {"__iadd__": "add", "__isub__": "subtract", "__imul__": "multiply",
            "__itruediv__": "divide"}[op]
    rng = np.random.default_rng(abs(hash((op, str(key)))) % 2**32)
    base = _words(rng, (5, 11, 13))
    for label, other in _operands(cp, rng, cp.asarray(base)[key].shape if cp.asarray(base)[key].ndim == 3
                                  else (1,) + cp.asarray(base)[key].shape):
        a = cp.asarray(base)
        b = cp.asarray(base)
        va, vb = a[key], b[key]
        if isinstance(other, cp.ndarray) and other.ndim > va.ndim:
            other = other[0]
        try:
            np.broadcast_shapes(va.shape, getattr(other, "shape", ()))
        except ValueError:
            continue
        if np.broadcast_shapes(va.shape, getattr(other, "shape", ())) != va.shape:
            continue
        vb._write(operand_values(UFUNCS[name](vb, other), vb.dtype))   # the general path
        assert _write.arith_in_place(name, va, other), label
        got, want = a.get().view(np.uint32), b.get().view(np.uint32)
        assert np.array_equal(got, want), (label, int((got != want).sum()))


def test_operator_takes_the_fused_kernel(cp, monkeypatch):
    from woof.backend.facade import _write
    calls = []
    real = _write.arith_in_place
    monkeypatch.setattr(_write, "arith_in_place", lambda *a: calls.append(a[0]) or real(*a))
    a = cp.ones((4, 6), dtype=cp.float32)
    a[1:3, 2:] += cp.full((2, 4), 2.0, dtype=cp.float32)
    a *= 3
    assert calls == ["add", "multiply"]
    assert np.array_equal(a.get(), np.array([[3] * 6, [3, 3, 9, 9, 9, 9], [3, 3, 9, 9, 9, 9], [3] * 6],
                                            dtype=np.float32))


def test_shared_buffer_keeps_the_general_path(cp):
    from woof.backend.facade import _write
    host = np.arange(24, dtype=np.float32)
    a = cp.asarray(host)
    assert not _write.arith_in_place("add", a[1:], a[:-1])
    a[1:] += a[:-1]
    host[1:] += host[:-1].copy()
    assert np.array_equal(a.get(), host)


def test_other_dtypes_keep_the_general_path(cp):
    from woof.backend.facade import _write
    a = cp.zeros(8, dtype=cp.float32)
    assert not _write.arith_in_place("add", a, cp.ones(8, dtype=cp.int32))
    assert not _write.arith_in_place("add", a, np.float64(1.0))
    assert not _write.arith_in_place("power", a, 2.0)
    assert not _write.arith_in_place("add", cp.zeros(8, dtype=cp.int32), 1)


def test_zero_stride_destination_keeps_the_general_path(cp):
    from woof.backend.facade import _write
    a = cp.broadcast_to(cp.zeros(1, dtype=cp.float32), (8,))
    assert not _write.arith_in_place("add", a, 1.0)
    b = cp.broadcast_to(cp.zeros((1, 4), dtype=cp.float32), (3, 4))
    assert not _write.arith_in_place("multiply", b, cp.ones((3, 4), dtype=cp.float32))
