"""WOOF_METAL_GPU_TIMES: each kernel's launches and GPU seconds, written at exit."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_SCRIPT = r'''
import numpy as np
import woof, cupy as cp
k = cp.RawKernel(r"""
extern "C" __global__ void twice(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f;
}""", "twice")
x = cp.ones(1 << 16, dtype=cp.float32)
for _ in range(7):
    k((256,), (256,), (x, np.int32(1 << 16)))
assert float(x.get()[3]) == 128.0
'''


def test_gpu_times_count_each_kernels_launches(tmp_path):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    out = tmp_path / "times-{pid}.json"
    env = dict(os.environ, WOOF_METAL_GPU_TIMES=str(out))
    subprocess.run([sys.executable, "-c", _SCRIPT], env=env, check=True, timeout=600)
    written = [json.loads(p.read_text()) for p in tmp_path.glob("times-*.json")]
    rows = [w["twice"] for w in written if "twice" in w]
    assert len(rows) == 1
    assert rows[0]["launches"] == 7
    assert rows[0]["gpu_seconds"] > 0
