"""In-place writes through the Metal facade against NumPy, word for word.

Every write into a facade view (``a[...] = v``, ``fill``, ``copyto``) runs
the strided copy kernel of woof/backend/facade/_write.py: one value for
every element takes the fill kernel, anything else the copy; the view's
axes are collapsed host-side where they are contiguous.  The same views
and values written into NumPy arrays are the reference.
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


def test_collapsed_walk_matches_every_view():
    from woof.backend.facade._write import _collapsed
    rng = np.random.default_rng(3)
    for _ in range(2000):
        nd = int(rng.integers(1, 5))
        a = np.zeros([int(x) for x in rng.integers(1, 7, nd)], dtype=np.int64)
        key = tuple(slice(int(rng.integers(0, 2)), None, int(rng.choice([1, 2, -1])))
                    if rng.random() < 0.5 else slice(None) for _ in a.shape)
        v = a[key]
        if rng.random() < 0.3:
            v = v.transpose()
        if v.size == 0:
            continue
        strides = [s // 8 for s in v.strides]
        shape, steps = _collapsed(v.shape, strides)
        got = []
        for i in range(v.size):
            rem, off = i, 0
            for d in range(len(shape) - 1, 0, -1):
                off += (rem % shape[d]) * steps[d]
                rem //= shape[d]
            got.append(off + rem * steps[0])
        assert got == [sum(i * s for i, s in zip(idx, strides)) for idx in np.ndindex(v.shape)]


_KEYS = [
    (Ellipsis,),
    (slice(None), slice(None), slice(0, 1)),
    (slice(None), slice(-2, None), slice(None)),
    (slice(None), slice(2, -2), slice(-3, None)),
    (slice(1, None, 2), slice(None, None, 3), slice(None)),
    (2, slice(None), slice(4, 9)),
]


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64, np.uint8, np.int16])
@pytest.mark.parametrize("key", _KEYS)
def test_writes_match_numpy(cp, dtype, key):
    rng = np.random.default_rng(11)
    host = (rng.random((5, 7, 9)) * 100).astype(dtype)
    dev = cp.asarray(host)
    want = host.copy()
    # one value for every element: the fill kernel
    want[key] = dtype(3)
    dev[key] = dtype(3)
    np.testing.assert_array_equal(dev.get(), want)
    # an array of values: the copy kernel
    vals = (rng.random(want[key].shape) * 50).astype(dtype)
    want[key] = vals
    dev[key] = cp.asarray(vals)
    np.testing.assert_array_equal(dev.get(), want)
    # a row broadcast over the view
    row = (rng.random(want[key].shape[-1:]) * 7).astype(dtype)
    want[key] = row
    dev[key] = cp.asarray(row)
    np.testing.assert_array_equal(dev.get(), want)
    dev.fill(dtype(1))
    assert (dev.get() == 1).all()


def test_fill_keeps_the_buffer_and_words(cp):
    a = cp.zeros((4, 6, 8), dtype=cp.float32)
    ptr = a.data.ptr
    a[:, :, :2] = np.float32(-0.0)
    a[1:3] = np.float32(np.nan)
    host = a.get()
    assert a.data.ptr == ptr
    assert (host[:, :, :2].view(np.uint32)[[0, 3]] == 0x80000000).all()
    assert np.isnan(host[1:3]).all()
    assert (host[[0, 3], :, 2:] == 0).all()
