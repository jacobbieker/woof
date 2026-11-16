"""``MemoryPointer.copy_from_device(_async)`` between two storages is a
device copy on the Metal facade (woof/backend/facade/_array.py,
``_DEVICE_COPY``).

The wrfout writer snapshots each history field this way
(woof/io/wrfout.py ``_device_copy``).  The copy used to go through the host,
reading the source back (a wait for every kernel queued before it) and then
writing it again.  Now it is one write the facade orders like any kernel:
it must see the writes queued before it, must not see writes queued after
it, must copy the bytes for any dtype and byte offset, and must not
synchronize.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_BUMP = r'''extern "C" __global__ void bump(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; for (int k = 0; k < 32; ++k) v = v * 0.5f + s; x[i] = v; } }'''


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


@pytest.fixture
def synchronizes(monkeypatch):
    import mlx.core as mx
    calls = []
    for name in ("synchronize", "eval"):
        fn = getattr(mx, name)

        def counted(*a, _fn=fn, _name=name, **k):
            calls.append(_name)
            return _fn(*a, **k)
        monkeypatch.setattr(mx, name, counted)
    return calls


def _host_bump(v, s):
    for _ in range(32):
        v = (v * np.float32(0.5) + np.float32(s)).astype(np.float32)
    return v


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint8, np.bool_, np.float16])
@pytest.mark.parametrize("device", ["1", "0"])
def test_device_copies_the_bytes(cp, monkeypatch, dtype, device):
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_DEVICE_COPY", device == "1")
    rng = np.random.default_rng(3)
    h = (rng.standard_normal((4, 9, 11)) * 50).astype(dtype)
    src = cp.asarray(h)
    dst = cp.asarray(np.zeros_like(h))
    dst.data.copy_from_device_async(src.data, src.nbytes)
    assert np.array_equal(dst.get().view(np.uint8), h.view(np.uint8))
    # part of a row, at byte offsets into both allocations
    dst2 = cp.asarray(np.zeros_like(h))
    n = 5 * h.itemsize
    dst2[2].data.copy_from_device(src[1, 3:].data, n)
    want = np.zeros_like(h)
    want[2].reshape(-1)[:5] = h[1, 3:].reshape(-1)[:5]
    assert np.array_equal(dst2.get().view(np.uint8), want.view(np.uint8))


def test_device_copy_is_ordered_without_a_synchronize(cp, monkeypatch, synchronizes):
    from woof.backend.facade import _array
    monkeypatch.setattr(_array, "_DEVICE_COPY", True)
    bump = cp.RawKernel(_BUMP, "bump")
    n = 3 * 40 * 50
    rng = np.random.default_rng(9)
    src_h = rng.standard_normal(n).astype(np.float32)
    src = cp.asarray(src_h)
    snaps = [cp.zeros(n, cp.float32) for _ in range(6)]
    grid = ((n + 255) // 256,)
    del synchronizes[:]
    want = []
    for k, snap in enumerate(snaps):
        bump(grid, (256,), (src, np.float32(k), np.int32(n)))
        src_h = _host_bump(src_h, k)
        snap.data.copy_from_device_async(src.data, src.nbytes)
        want.append(src_h.copy())
        bump(grid, (256,), (src, np.float32(-k), np.int32(n)))     # after the copy
        src_h = _host_bump(src_h, -k)
    assert synchronizes == [], synchronizes
    for snap, ref in zip(snaps, want):
        assert np.array_equal(snap.get().view(np.uint32), ref.view(np.uint32))
    assert np.array_equal(src.get().view(np.uint32), src_h.view(np.uint32))


def test_wrfout_device_copy_snapshot(cp):
    from woof.io.wrfout import _device_copy
    h = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    src = cp.asarray(h)
    dst = cp.empty_like(src)
    stream = cp.cuda.Stream(non_blocking=True)
    _device_copy(cp, dst, src, stream)
    src += cp.float32(1.0)
    stream.synchronize()
    assert np.array_equal(dst.get(), h)


def test_a_copy_into_a_page_locked_block_is_complete_on_return(cp):
    """hostAlloc memory reached through its device address: the host reads
    it straight after the copy, so that copy still completes on return."""
    from woof.backend.facade import cuda as fcuda
    n = 1 << 22
    src_h = np.arange(n, dtype=np.float32)
    src = cp.asarray(src_h)
    bump = cp.RawKernel(_BUMP, "bump")
    for k in range(4):                    # keep the GPU busy past the copy's return
        bump(((n + 255) // 256,), (256,), (src, np.float32(k), np.int32(n)))
        src_h = _host_bump(src_h, k)
    ptr = cp.cuda.runtime.hostAlloc(src.nbytes, 0)
    try:
        dev = fcuda._pointer_attributes(ptr).devicePointer
        block = cp.ndarray((n,), cp.float32,
                           memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(dev, src.nbytes, None), 0))
        block.data.copy_from_device(src.data, src.nbytes)
        host = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_float * n).from_address(ptr))
        assert np.array_equal(host.view(np.uint32), src_h.view(np.uint32))
    finally:
        cp.cuda.runtime.freeHost(ptr)
