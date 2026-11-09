"""A constant index into a long private array reads the element, on any GPU.

Apple's shader compiler (macOS 26.5, M3 Max) read ``r[0]`` of a function-
local array of more than 32 elements as 0 once loops of runtime trip count
had filled and read the array -- in a plain ``mx.fast.metal_kernel`` too.
NSSL-2's rain sedimentation exported ``rho[0] * flux`` through it as 0.  The
translator hides such constant indices from the compiler
(``__woof_opaque_index``); the kernel's words are CUDA's.
"""

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

_SOURCE = r'''
extern "C" __global__ void after_loops(const float* rho_in, float* q, float* out, int nz, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float rho[64];
    for (int k = 0; k < nz; ++k) rho[k] = rho_in[k * n + c];
    float t = 0.0f;
    for (int k = 0; k < nz; ++k) t += rho[k];
    q[c] = t;
    out[c] = rho[0] + 2.0f * rho[5];
}
'''


def test_a_constant_index_after_runtime_loops_reads_the_element():
    import cupy as cp
    n, nz = 6, 12
    rho = (1.2 + 0.01 * np.arange(nz * n)).astype(np.float32).reshape(nz, n)
    q, out = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
    cp.RawKernel(_SOURCE, "after_loops")((1,), (64,), (cp.asarray(rho), q, out, np.int32(nz), np.int32(n)))
    assert cp.asnumpy(out).tobytes() == (rho[0] + np.float32(2.0) * rho[5]).astype(np.float32).tobytes()
    assert np.allclose(cp.asnumpy(q), rho.sum(axis=0, dtype=np.float32), rtol=1e-6)
