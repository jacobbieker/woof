"""The Metal and host facades write one ``ElementwiseKernel`` text.

Both ``woof.backend.facade`` and ``woof.backend.hostfacade`` take their
parameter parsing and kernel source from ``woof.backend._elementwise_source``.
The texts below are the ones the two copies wrote before they were merged;
a change to either facade's text now shows here, for both.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.backend import _elementwise_source as shared
from woof.backend.facade import _elementwise as metal
from woof.backend.hostfacade import _elementwise as host

ELEMENTWISE = (
    'typedef double T;\n\nextern "C" __global__ void k_gen(const double* __woof_in_x, double y, '
    'double* __woof_out_z, long long __woof_n) {\n    int q=0;\n'
    '    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n'
    '    if (i >= __woof_n) return;\n    const double x = __woof_in_x[i];\n'
    '    double& z = __woof_out_z[i];\n    z = x > y ? x : y;\n}\n')
REDUCTION = (
    'typedef float T;\n\nextern "C" __global__ void sum(const float* __woof_in_x, float* __woof_out, '
    'long long __woof_n, long long __woof_r) {\n'
    '    long long __woof_o = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n'
    '    if (__woof_o >= __woof_n) return;\n    float a = (float)(0);\n'
    '    for (long long __woof_k = 0; __woof_k < __woof_r; ++__woof_k) {\n'
    '        const float x = __woof_in_x[__woof_o * __woof_r + __woof_k];\n'
    '        float b = (float)(x);\n        a = (float)(a + b);\n    }\n'
    '    float y;\n    y = a;\n    __woof_out[__woof_o] = y;\n}\n')


@pytest.mark.parametrize("module", [metal, host], ids=["metal", "host"])
def test_elementwise_text(module):
    kernel = module.ElementwiseKernel("T x, T y", "T z", "z = x > y ? x : y", name="k gen",
                                      loop_prep="int q=0;")
    assert isinstance(kernel, shared.ElementwiseSource)
    assert kernel._source({"T": np.float64}, (True, False)) == ELEMENTWISE
    assert [p.is_const for p in kernel.params] == [True, True, False]


@pytest.mark.parametrize("module", [metal, host], ids=["metal", "host"])
def test_reduction_text(module):
    kernel = module.ReductionKernel("T x", "T y", "x", "a + b", "y = a", "0", name="sum")
    assert isinstance(kernel, shared.ReductionSource)
    assert kernel._source({"T": np.float32}, (True,)) == REDUCTION
    with pytest.raises(ValueError, match="one output"):
        module.ReductionKernel("raw T x", "T y", "x", "a + b", "y = a", "0")
