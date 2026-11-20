"""A host upload on the Metal facade is held as it was built, word for word.

``cp.asarray(host)`` used to store ``mx.contiguous`` of the uploaded array.
That is a lazy primitive on the GPU stream, so the first evaluation of a new
upload waited for every launch queued ahead of it.  A device address table
is read at its launch (woof/backend/metal/loader.py, _device_table_words),
so each step's freshly built table drained the queue: about 14 ms a step on
the hex Edinburgh variable-resolution mesh (2026-11-16).  The upload is now
the flat array built from host memory (woof/backend/facade/_array.py,
``_READY_UPLOADS``; WOOF_METAL_READY_UPLOADS=0 restores the old path).
The words, shape, strides and dtype must be what the old path gave.
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


DTYPES = (np.float32, np.float64, np.float16, np.int8, np.int16, np.int32, np.int64,
          np.uint8, np.uint16, np.uint32, np.uint64, np.bool_, np.complex64, np.complex128)


def _host(dtype, shape, seed=7):
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 256, size=int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize,
                         dtype=np.uint8)
    host = words.view(dtype).reshape(shape).copy()
    if np.dtype(dtype) == np.bool_:
        host = host.view(np.uint8) % 2 == 1
    return host


def _upload(cp, monkeypatch, ready, make):
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_READY_UPLOADS", ready)
    return make()


def _bytes(x):
    return np.ascontiguousarray(x).reshape(-1).view(np.uint8).tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (1,), (6,), (11, 5), (3, 4, 7)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_an_upload_keeps_the_old_paths_words(cp, monkeypatch, dtype, shape, order):
    host = _host(dtype, shape)
    for maker in (lambda: cp.asarray(np.asarray(host, order=order)),
                  lambda: cp.array(host, order=order)):
        old = _upload(cp, monkeypatch, False, maker)
        new = _upload(cp, monkeypatch, True, maker)
        assert new.dtype == old.dtype and new.shape == old.shape and new.strides == old.strides
        assert _bytes(new.get()) == _bytes(old.get()) == _bytes(host)
        assert new._st.arr.shape == old._st.arr.shape
        assert new._st.arr.dtype == old._st.arr.dtype


def test_an_upload_is_not_a_lazy_copy(cp, monkeypatch):
    """The ready path never calls _materialize (mx.contiguous) on an upload."""
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_READY_UPLOADS", True)

    def refuse(arr):
        raise AssertionError("an upload went through mx.contiguous")

    monkeypatch.setattr(_array, "_materialize", refuse)
    for dtype in DTYPES:
        for host in (_host(dtype, (6,)), _host(dtype, (4, 5)), np.asfortranarray(_host(dtype, (4, 5)))):
            cp.asarray(host)


def test_an_upload_is_its_own_copy(cp, monkeypatch):
    """Later host writes do not reach the device array, nor device writes the host."""
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_READY_UPLOADS", True)
    host = np.arange(12, dtype=np.float32)
    dev = cp.asarray(host)
    host[:] = -1
    assert dev.get().tolist() == list(range(12))
    dev[...] = 5
    assert host.tolist() == [-1.0] * 12
    assert dev.get().tolist() == [5.0] * 12


def test_an_upload_is_written_in_place_like_any_storage(cp, monkeypatch):
    """A kernel writing through the upload's address sees and keeps its words."""
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_READY_UPLOADS", True)
    a = cp.asarray(np.arange(8, dtype=np.float32))
    ptr = a.data.ptr
    a += 1
    a[2:4] = 0
    assert a.data.ptr == ptr
    assert a.get().tolist() == [1, 2, 0, 0, 5, 6, 7, 8]


def test_an_empty_upload_takes_the_general_path(cp, monkeypatch):
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_READY_UPLOADS", True)
    for dtype in (np.float32, np.float64, np.complex128):
        out = cp.asarray(np.zeros((0, 3), dtype=dtype))
        assert out.shape == (0, 3) and out.dtype == dtype
