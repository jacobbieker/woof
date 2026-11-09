"""The compile prefetch: a run records its units; the next builds them ahead, the same."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_SCRIPT = r'''
import json, sys, time
import numpy as np
import woof, cupy as cp
from woof.backend.metal import prefetch
src = """extern "C" __global__ void scale(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * s + 1.0f;
}
extern "C" __global__ void shift(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] - 0.25f;
}"""
first = cp.RawModule(code="// unit one\n" + src)
second = cp.RawModule(code="// unit two\n" + src)
deadline = time.monotonic() + 120
while prefetch.status()["queued"] and sum(prefetch.status()[k] for k in ("built", "failed")) < prefetch.status()["queued"]:
    if time.monotonic() > deadline:
        break
    time.sleep(0.1)
status = prefetch.status()
x = cp.asarray(np.linspace(0, 1, 1000, dtype=np.float32))
for m in (first, second):
    m.get_function("scale")((4,), (256,), (x, np.float32(1.5), np.int32(1000)))
    m.get_function("shift")((4,), (256,), (x, np.int32(1000)))
print(json.dumps({"status": status, "words": x.get().view(np.uint32).tolist()}))
'''


def _run(env):
    out = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, check=True, timeout=600,
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_a_second_run_builds_the_first_runs_units_ahead_with_the_same_words(tmp_path):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    env = dict(os.environ, WOOF_METAL_PLAN_DIR=str(tmp_path / "plans"))
    first = _run(env)
    assert first["status"]["queued"] == 0                 # nothing recorded yet
    plans = list((tmp_path / "plans").glob("*.json"))
    assert len(plans) == 1
    rows = json.loads(plans[0].read_text())
    assert sum(1 for r in rows if set(r["kernels"]) == {"scale", "shift"}) == 2
    second = _run(env)
    assert second["status"]["queued"] >= 2
    assert second["status"]["built"] >= 2 and second["status"]["failed"] == 0
    assert second["words"] == first["words"]
    off = _run(dict(env, WOOF_METAL_PREFETCH_WORKERS="0"))
    assert off["status"]["queued"] == 0 and off["words"] == first["words"]
