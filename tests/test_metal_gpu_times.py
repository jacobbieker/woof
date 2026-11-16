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


_ISOLATE_SCRIPT = r'''
import hashlib
import numpy as np
import woof, cupy as cp
k = cp.RawKernel(r"""
extern "C" __global__ void twice(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f;
}""", "twice")
n = 1 << 16
x = cp.ones(n, dtype=cp.float32)
y = cp.arange(n, dtype=cp.float32)
for _ in range(3):
    k(((n + 255) // 256,), (256,), (x, np.int32(n)))
    x[...] = cp.sqrt(y * x + 1.0) * 0.5 + x   # WRITE-SITE
    x[:7] = 0                                 # FILL-SITE
print(hashlib.sha256(x.get().tobytes()).hexdigest())
'''


def _run_times(tmp_path, tag, isolate):
    script = tmp_path / "isolate_script.py"
    script.write_text(_ISOLATE_SCRIPT, encoding="utf-8")
    out = tmp_path / f"{tag}-{{pid}}.json"
    env = dict(os.environ, WOOF_METAL_GPU_TIMES=str(out), WOOF_METAL_GPU_TIMES_ISOLATE=isolate)
    done = subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=600,
                          capture_output=True, text=True)
    (written,) = [json.loads(p.read_text()) for p in tmp_path.glob(f"{tag}-*.json")]
    return done.stdout.split()[-1], written


def test_isolated_gpu_times_key_writes_by_call_site_and_keep_the_words(tmp_path):
    """WOOF_METAL_GPU_TIMES_ISOLATE: the lazy right-hand side of a write is
    counted apart from the copy, writes are keyed by their call site, and
    the words are those of an ordinary run."""
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    words, plain = _run_times(tmp_path, "plain", "0")
    words_isolated, isolated = _run_times(tmp_path, "isolated", "1")
    assert words_isolated == words
    assert plain["twice"]["launches"] == 3 and plain["woof_copy_4"]["launches"] == 3
    lines = _ISOLATE_SCRIPT.splitlines()
    write_line = next(i for i, line in enumerate(lines, 1) if "WRITE-SITE" in line)
    fill_line = next(i for i, line in enumerate(lines, 1) if "FILL-SITE" in line)
    copy = [k for k in isolated if k.startswith("woof_copy_4@") and k.endswith(f"isolate_script.py:{write_line}")]
    fill = [k for k in isolated if k.startswith("woof_fill_4@") and k.endswith(f"isolate_script.py:{fill_line}")]
    rhs = [k for k in isolated if k.startswith("mlx@") and k.endswith(f"isolate_script.py:{write_line}")]
    assert len(copy) == len(fill) == len(rhs) == 1, sorted(isolated)
    assert isolated[copy[0]]["launches"] == isolated[fill[0]]["launches"] == 3
    assert isolated["twice"]["launches"] == 3
    assert isolated[rhs[0]]["gpu_seconds"] > 0
    assert all(row["min_ms"] <= row["max_ms"] for row in isolated.values())
