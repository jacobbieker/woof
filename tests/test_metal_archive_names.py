"""A unit with a long name still archives its pipelines, under a bounded file name.

A unit's library key carries its integer defines (SSiB with its eight table
digests: 290 characters), and the archive's file name was the whole key: past
the file system's 255-byte limit, -[MTLBinaryArchive serializeToURL:] crashed
the process (SIGSEGV in objc_retain inside airntSerializeToURL).  Names past
120 characters are now cut to 96 plus a digest of the whole key.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_SCRIPT = r'''
import numpy as np
import woof, cupy as cp
from woof.backend.metal import loader
src = """extern "C" __global__ void plus_two(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 2.0f;
}"""
name = "longunit[" + ",".join(f"DEFINE_NUMBER_{i}={1000000007 * (i + 1)}" for i in range(12)) + "]"
for variant in (name, name.replace("DEFINE", "OTHER")):
    fn = loader.load_unit(variant, src).get_function("plus_two")
    x = cp.zeros(256, dtype=cp.float32)
    fn((1,), (256,), (x, np.int32(256)))
    assert float(x.get()[3]) == 2.0
'''


def test_long_unit_names_archive_under_bounded_distinct_names(tmp_path):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    pipelines = tmp_path / "pipelines"
    env = dict(os.environ, WOOF_METAL_PIPELINE_DIR=str(pipelines))
    subprocess.run([sys.executable, "-c", _SCRIPT], env=env, check=True, timeout=600)
    names = [p.name for p in pipelines.glob("*plus_two.metalarchive")] or \
        [p.name for p in pipelines.glob("woof_longunit*.metalarchive")]
    assert len(names) == 2, sorted(p.name for p in pipelines.iterdir())
    assert all(len(n) < 160 for n in names)
    assert names[0] != names[1]
