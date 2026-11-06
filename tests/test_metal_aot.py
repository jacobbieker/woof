"""Compiled units persist across processes (woof.backend.metal.dylib).

* :func:`split` cuts a translated unit into a body (kernels as ordinary
  functions behind opaque-typed wrappers) and a thin library of forwarding
  kernels with the original bindings; a unit that declares threadgroup
  memory is left whole.
* On the device, a unit is built once as a serialized dynamic library; a
  second process loads it from disk, compiles only the thin library, and
  computes the same words as the whole-unit build.
* Translations persist too, keyed by the preprocessed text.
"""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_UNIT = r'''
struct Pair { float a; float b; };
__device__ float mix(Pair p, float x) { return p.a * x + p.b; }
extern "C" __global__ void affine(const float* x, float* y, Pair p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = mix(p, x[i]);
}
extern "C" __global__ void twice(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] * 2.0f;
}
'''

_SHARED = r'''
extern "C" __global__ void tile(const float* x, float* y) {
    __shared__ float s[64];
    s[threadIdx.x] = x[threadIdx.x];
    __syncthreads();
    y[threadIdx.x] = s[63 - threadIdx.x];
}
'''


def test_split_cuts_a_unit_into_body_and_thin_kernels():
    from woof.backend.metal.dylib import split
    from woof.backend.translate import translate
    body, thin = split(translate(_UNIT, unit="affine").msl)
    assert "[[kernel]]" not in body.split("// ---- translated unit: ", 1)[1]
    for name in ("affine", "twice"):
        assert f"void __woof_body_{name}(" in body
        assert f"[[kernel]] void {name}(" in thin
        assert f"extern void __woof_body_{name}(" in thin
    assert "[[buffer(0)]]" in thin and "[[thread_position_in_threadgroup]]" in thin
    assert "Pair" not in thin                      # opaque: the thin library needs no unit types


def test_a_unit_with_threadgroup_memory_stays_whole():
    from woof.backend.metal.dylib import split
    from woof.backend.translate import translate
    assert split(translate(_SHARED, unit="tile").msl) is None


_RUN = r'''
import json, sys
import numpy as np
import cupy as cp
src = sys.argv[1]
mod = cp.RawModule(code=src)
x = cp.asarray(np.linspace(-3, 3, 100, dtype=np.float32))
y = cp.zeros(100, dtype=cp.float32)
pair = np.array([1.25, -0.5], dtype=np.float32).view(np.dtype([("a", np.float32), ("b", np.float32)]))[0]
mod.get_function("affine")((1,), (128,), (x, y, pair, np.int32(100)))
mod.get_function("twice")((1,), (128,), (y, np.int32(100)))
print(json.dumps(cp.asnumpy(y).view(np.uint32).tolist()))
'''


def _run(tmp_path, libraries, source):
    env = dict(os.environ, WOOF_METAL_LIBRARY_DIR=str(libraries), WOOF_METAL_COMPILE_LOG="1",
               WOOF_METAL_TRANSLATION_DIR=str(tmp_path / "translations"),
               WOOF_METAL_PIPELINE_DIR=str(tmp_path / "pipelines"))
    done = subprocess.run([sys.executable, "-c", _RUN, source], env=env, capture_output=True,
                          text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def test_a_second_process_loads_the_unit_from_disk_with_the_same_words(tmp_path):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    # a nonce keeps the OS shader cache and any earlier run out of the measurement
    source = _UNIT + f"\n// {tmp_path.name}\n"
    whole, _ = _run(tmp_path, "off", source)
    first, log1 = _run(tmp_path, tmp_path / "libraries", source)
    second, log2 = _run(tmp_path, tmp_path / "libraries", source)
    assert "dynamic library (built)" in log1
    assert "dynamic library (disk)" in log2 and "dynamic library (built)" not in log2
    assert first == second == whole
    x = np.linspace(-3, 3, 100, dtype=np.float32)
    want = ((np.float32(1.25) * x + np.float32(-0.5)) * np.float32(2.0)).view(np.uint32).tolist()
    assert whole == want
    assert len(list((tmp_path / "libraries").glob("*.metallib"))) == 1
    assert list((tmp_path / "translations").glob("*.pickle"))


def test_a_translation_is_read_back_unchanged(tmp_path, monkeypatch):
    from woof.backend import translate as tr
    monkeypatch.setenv(tr.TRANSLATION_DIR_ENV, str(tmp_path))
    first = tr.translate(_UNIT, unit="affine")
    assert len(list(tmp_path.glob("*.pickle"))) == 1
    second = tr.translate(_UNIT, unit="affine")
    assert second.msl == first.msl and second.key == first.key
    assert second.kernels == first.kernels
    # another unit name, or another integer macro, is another entry
    tr.translate(_UNIT, unit="affine2")
    tr.translate(_UNIT, unit="affine", defines=("KMAX=8",))
    assert len(list(tmp_path.glob("*.pickle"))) == 3
