"""Mapped page-locked host memory on the MLX backend (WOOF_BACKEND=mlx).

CUDA code (tilestream's firebrand owner) allocates host memory with
``cudaHostAlloc``, finds its device address with
``cudaPointerGetAttributes`` and wraps it with ``UnownedMemory``, so kernels
work on the host bytes without a device copy.  Apple memory is unified, and
the facade's hostAlloc block is one Metal buffer seen both ways.  Breakage
it prevents: the WRF fire-spotting driver refused with "mapped firebrand
fields require page-locked host bytes for device access".
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_ADD_INDEX = r'''
extern "C" __global__ void add_index(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += (float)i;
}
'''


def test_mapped_host_bytes_are_the_device_array():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy as cp

    n = 64
    ptr = cp.cuda.runtime.hostAlloc(n * 4, 0)
    try:
        attributes = cp.cuda.runtime.pointerGetAttributes(ptr)
        assert attributes.type == cp.cuda.runtime.memoryTypeHost
        assert attributes.devicePointer
        host = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_float * n).from_address(ptr))
        host[:] = np.arange(n, dtype=np.float32) * 0.5
        memory = cp.cuda.UnownedMemory(attributes.devicePointer, n * 4, None)
        device = cp.ndarray((n,), cp.float32, cp.cuda.MemoryPointer(memory, 0))
        # the device array reads the host's words, and its kernel writes land in them
        assert np.array_equal(cp.asnumpy(device), host)
        cp.RawKernel(_ADD_INDEX, "add_index")((1,), (n,), (device, np.int32(n)))
        cp.cuda.runtime.deviceSynchronize()
        want = np.arange(n, dtype=np.float32) * 1.5
        assert np.array_equal(host, want)
        # a facade write is in place too
        device[: n // 2] = 0.0
        cp.cuda.runtime.deviceSynchronize()
        assert not host[: n // 2].any() and np.array_equal(host[n // 2:], want[n // 2:])
        # an ordinary device array is device memory
        other = cp.zeros(4, cp.float32)
        assert cp.cuda.runtime.pointerGetAttributes(other.data.ptr).type == cp.cuda.runtime.memoryTypeDevice
    finally:
        cp.cuda.runtime.freeHost(ptr)
