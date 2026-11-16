"""``ndarray.get(out=...)`` on the Metal facade copies once, word for word.

On unified memory an evaluated MLX buffer is host memory, so ``get(out=)``
reads it in place and copies straight into ``out`` (woof/backend/facade/
_array.py, ``_DIRECT_OUT``).  It used to copy into a fresh host array and
then into ``out``: the history writers stage every field of a frame this
way, into page-locked blocks (alloc_pinned_memory), and the extra copy and
allocation about doubled a 40-field Edinburgh 3 km frame's staging time
(2026-11-16).  The words, the dtype ``out`` keeps and the returned object
must be what the two-copy path gave, for every view and dtype.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


DTYPES = (np.float32, np.float64, np.float16, np.int32, np.int64, np.uint8, np.bool_,
          np.complex64, np.complex128)


def _bytes(x):
    return np.ascontiguousarray(x).reshape(-1).view(np.uint8)


def _views(a, h):
    yield a, h
    yield a[:, ::2, 1:], h[:, ::2, 1:]
    yield a.transpose(2, 0, 1), h.transpose(2, 0, 1)
    yield a[1], h[1]
    yield a[2, 3, 4], np.asarray(h[2, 3, 4])
    if h.dtype != np.bool_:
        yield a.view(np.uint8), h.view(np.uint8)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("direct", ["1", "0"])
def test_get_out_matches_the_host_words(cp, monkeypatch, dtype, direct):
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_DIRECT_OUT", direct == "1")
    rng = np.random.default_rng(7)
    h = (rng.standard_normal((5, 6, 7)) * 100).astype(dtype)
    h.reshape(-1)[:3] = (np.nan, np.inf, -0.0) if h.dtype.kind in "fc" else h.reshape(-1)[:3]
    a = cp.asarray(h)
    for view, want in _views(a, h):
        out = np.empty(want.shape, want.dtype)
        assert view.get(out=out) is out
        assert np.array_equal(_bytes(out), _bytes(want))
        if want.dtype.kind not in "cb":
            # a wider ``out`` takes the values, cast as NumPy casts them
            wide = np.empty(want.shape, np.float64)
            view.get(out=wide)
            assert np.array_equal(wide, want.astype(np.float64), equal_nan=True)


def test_get_out_into_a_pinned_block_sees_the_latest_write(cp):
    """The writers' pattern: a page-locked block, a field written in place
    since the last frame, then get(out=block)."""
    field = cp.asarray(np.arange(4096, dtype=np.float32))
    memory = cp.cuda.alloc_pinned_memory(field.nbytes)
    host = np.frombuffer(memory, dtype=np.float32, count=field.size)
    for step in range(4):
        field += cp.float32(1.5)
        field.get(out=host, blocking=False)
        assert np.array_equal(host, np.arange(4096, dtype=np.float32) + np.float32(1.5) * (step + 1))
