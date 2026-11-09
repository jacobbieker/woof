"""A facade storage's GPU address is known after its first evaluation.

An in-place launch's output shares its input's buffer (rebind_written), so
the address stands; only rebind() (a new buffer) forgets it.  Asking again
used to evaluate the storage each time -- a host sync on every launch still
writing it: the microphysics ring copy's table key asked for 390 addresses
per model step.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_ADD = r'''
extern "C" __global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
'''


def test_the_address_survives_in_place_launches_without_evaluating(monkeypatch):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy as cp
    import mlx.core as mx
    from woof.backend.metal import ext

    x = cp.zeros(4096, dtype=cp.float32)
    first = x.data.ptr
    kernel = cp.RawKernel(_ADD, "add_one")
    evaluated = []
    real_eval = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *a: (evaluated.append(a), real_eval(*a))[1])
    for _ in range(5):
        kernel((16,), (256,), (x, np.int32(4096)))
        assert x.data.ptr == first
    assert not evaluated
    monkeypatch.setattr(mx, "eval", real_eval)
    assert int(ext.load().gpu_address(x._st.arr)) == first
    assert float(x.get()[7]) == 5.0
    # a new buffer: the address is measured again
    st = x._st
    st.rebind(mx.zeros(4096, dtype=mx.float32))
    assert st._address is None
    assert x.data.ptr == int(ext.load().gpu_address(st.arr))
