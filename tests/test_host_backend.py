"""The host backend (WOOF_BACKEND=host): CUDA units built as host C++.

Breakage this prevents: a block's threads that synchronise must see each
other's shared-memory writes (fibers, woof/backend/host/rt.h), warp
collectives must exchange lanes, atomics must hold across blocks that run
on several workers at once, binary32 arithmetic must be CUDA's
--fmad=false words (nothing contracted) with subnormals flushed exactly in
-ftz=true units, and ``import cupy`` must hand out arrays that woof's
``isinstance(x, np.ndarray)`` checks treat as device arrays, as CuPy's.
Each probe runs in a fresh process, because the backend is chosen once per
process.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not any(shutil.which(c) for c in ("clang++", "g++", "c++")),
    reason="the host backend needs a C++ compiler")


def _run(tmp_path, body):
    env = dict(os.environ, WOOF_BACKEND="host", WOOF_HOST_CACHE=str(tmp_path / "cache"),
               WOOF_HOST_THREADS="4", CUDA_VISIBLE_DEVICES="-1")
    script = "import json, numpy as np, woof\nimport cupy as cp\nout = {}\n" + body + \
        "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                          timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_import_cupy_is_a_facade_over_host_memory(tmp_path):
    out = _run(tmp_path, r'''
x = cp.arange(10, dtype=cp.float32)
out["facade"] = bool(cp.__woof_facade__)
out["device_not_numpy"] = isinstance(x, cp.ndarray) and not isinstance(x, np.ndarray)
out["cuda_interface"] = hasattr(x, "__cuda_array_interface__")
s = (x * 2 + 1).sum()
out["reduction_0d"] = isinstance(s, cp.ndarray) and s.shape == () and float(s) == 100.0
out["dispatch"] = isinstance(np.where(x > 3, x, 0), cp.ndarray)
h = cp.asnumpy(x); h[0] = 99
out["asnumpy_copies"] = type(h) is np.ndarray and float(x[0]) == 0.0
v = x.reshape(2, 5); v[0, 1] = 42
out["views_share"] = float(x[1]) == 42.0 and x.data.mem is v.data.mem
m = cp.ndarray((10,), cp.float32, memptr=x.data)
out["memptr"] = float(m[1]) == 42.0
''')
    assert all(out.values()), out


def test_barriers_shared_memory_warps_and_atomics(tmp_path):
    out = _run(tmp_path, r'''
src = r"""
extern "C" __global__ void blocksum(const float* x, float* out, int n) {
  __shared__ float s[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? x[i] : 0.0f;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
extern "C" __global__ void warps(const float* x, float* sums, unsigned* ballots, int* count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = x[i];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  unsigned b = __ballot_sync(0xffffffffu, x[i] > 0.5f);
  if ((threadIdx.x & 31) == 0) { sums[i / 32] = v; ballots[i / 32] = b; }
  atomicAdd(count, 1);
}
extern "C" __global__ void reverse(float* out) {
  extern __shared__ float buf[];
  buf[threadIdx.x] = (float)(blockIdx.x * 1000 + threadIdx.x);
  __syncthreads();
  out[blockIdx.x * blockDim.x + threadIdx.x] = buf[blockDim.x - 1 - threadIdx.x];
}
"""
mod = cp.RawModule(code=src)
rng = np.random.default_rng(0)
x = rng.random(1000, dtype=np.float32)
sums = cp.zeros(4, cp.float32)
mod.get_function("blocksum")((4,), (256,), (cp.asarray(x), sums, np.int32(1000)))
ref = [float(np.add.reduce(np.pad(x, (0, 24))[b * 256:(b + 1) * 256])) for b in range(4)]
out["blocksum"] = np.allclose(sums.get(), ref, rtol=1e-6)
x2 = rng.random(512, dtype=np.float32)
ws, wb, n = cp.zeros(16, cp.float32), cp.zeros(16, cp.uint32), cp.zeros(1, cp.int32)
mod.get_function("warps")((4,), (128,), (cp.asarray(x2), ws, wb, n))
out["shuffle"] = np.allclose(ws.get(), x2.reshape(16, 32).sum(1), rtol=1e-6)
out["ballot"] = [int(b) for b in wb.get()] == [
    sum(1 << l for l in range(32) if x2[w * 32 + l] > 0.5) for w in range(16)]
out["atomics"] = int(n.get()[0]) == 512
r = cp.zeros(64 * 64, cp.float32)
mod.get_function("reverse")((64,), (64,), (r,), shared_mem=64 * 4)
expect = np.array([b * 1000 + 63 - t for b in range(64) for t in range(64)], np.float32)
out["dynamic_shared"] = bool((r.get() == expect).all())
''')
    assert all(out.values()), out


def test_binary32_words(tmp_path):
    out = _run(tmp_path, r'''
src = r"""
extern "C" __global__ void k(const float* a, const float* b, const float* c, float* r) {
  r[0] = a[0] * b[0] + c[0];          // never contracted
  r[1] = a[1] * b[1];                 // subnormal product
  float q; asm("mul.rn.f32 %0, %1, %2;" : "=f"(q) : "f"(a[1]), "f"(b[1])); r[2] = q;
  r[3] = fminf(-0.0f, 0.0f);
  r[4] = fmaxf(-0.0f, 0.0f);
  float m; asm("min.f32 %0, %1, %2;" : "=f"(m) : "f"(0.0f), "f"(c[1])); r[5] = m;
}
"""
a = np.array([1.1, 1e-20], np.float32); b = np.array([1.3, 1e-20], np.float32)
c = np.array([-(np.float32(1.1) * np.float32(1.3)), 1e-40], np.float32)
for label, opts in (("ftz", ("-std=c++17",)), ("keep", ("-std=c++17", "--ftz=false"))):
    r = cp.zeros(6, cp.float32)
    cp.RawModule(code=src, options=opts).get_function("k")((1,), (1,), (cp.asarray(a), cp.asarray(b), cp.asarray(c), r))
    out[label] = [float(v) for v in r.get()] + [bool(np.signbit(r.get()[3])), bool(np.signbit(r.get()[4]))]
''')
    sub = float(__import__("numpy").float32(1e-20) * __import__("numpy").float32(1e-20))
    for label in ("ftz", "keep"):
        assert out[label][0] == 0.0, out           # a*b+c with c = -(a*b): no FMA
        assert out[label][2] == sub, out           # PTX without .ftz keeps subnormals
        assert out[label][5] == 0.0, out           # PTX min.f32 orders a subnormal above +0
        assert out[label][6] is True and out[label][7] is False, out   # -0 < +0
    assert out["ftz"][1] == 0.0, out
    assert out["keep"][1] == sub, out


def test_launch_errors_are_raised_and_the_pool_recovers(tmp_path):
    out = _run(tmp_path, r'''
from woof.backend.host.loader import LaunchError
src = r"""
extern "C" __global__ void trap(int* x) { if (threadIdx.x == 3) __trap(); x[threadIdx.x] = 1; }
extern "C" __global__ void split(int* x) { if (threadIdx.x < 16) __syncthreads(); else __syncwarp(); x[0] = 1; }
extern "C" __global__ void ok(int* x) { x[blockIdx.x * blockDim.x + threadIdx.x] = 7; }
"""
mod = cp.RawModule(code=src)
for name in ("trap", "split"):
    try:
        mod.get_function(name)((2,), (32,), (cp.zeros(64, cp.int32),))
        out[name] = "no error"
    except LaunchError as exc:
        out[name] = str(exc)
y = cp.zeros(64, cp.int32)
mod.get_function("ok")((2,), (32,), (y,))
out["recovered"] = bool((y.get() == 7).all())
''')
    assert "__trap" in out["trap"], out
    assert "deadlock" in out["split"], out
    assert out["recovered"], out


def test_elementwise_and_reduction_kernels(tmp_path):
    out = _run(tmp_path, r'''
ek = cp.ElementwiseKernel("float32 a, float32 b", "float32 c", "c = a * b + 1.0f", "ew")
x = cp.arange(8, dtype=cp.float32)
out["elementwise"] = [float(v) for v in ek(x, x).get()]
rk = cp.ReductionKernel("T x", "T y", "x", "a + b", "y = a", "0", "rsum")
out["reduction"] = int(rk(cp.arange(100, dtype=cp.int32)))
''')
    assert out["elementwise"] == [float(i * i + 1) for i in range(8)], out
    assert out["reduction"] == 4950, out
