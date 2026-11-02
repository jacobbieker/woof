"""Translator constructs the hex lane's MPAS physics units brought in.

* Fortran's GOTO spaghetti, transcribed: forward jumps whose regions
  cross (MM5 sfclay's regime branches: ``IF (br < 0) GOTO 310``,
  ``IF (br < 0.2) GOTO 270`` ... ``GOTO 320``), and a forward jump out of a
  backward retry loop (Noah's frh2o: ``1001 IF (..) GOTO 1002`` ...
  ``GOTO 1001`` / ``1002``).  Both were refused as overlapping regions.
* Inline PTX on register bits: sign flips (``xor.b32``), bit moves, and
  NaN-ordered selects (``setp.lt.f32 p; selp.f32``) without ``.ftz``,
  which keep subnormals, signed zeros and NaN payloads.
* A string literal stored in a struct (MYNN's ``MynnText``) lives in
  ``constant`` memory.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


_CROSSING = r'''
extern "C" __global__ void crossing(const float* br, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = br[i];
    float r;
    if (b < 0.0f) goto L310;
    if (b < 0.2f) goto L270;
    r = 1.0f;
    goto L320;
L270:
    if (b == 0.0f) goto L280;
    r = 2.0f;
    goto L320;
L280:
    r = 3.0f;
    goto L320;
L310:
    r = 4.0f;
L320:
    out[i] = r * 10.0f + b;
}
'''


def test_crossing_forward_goto_regions(cp):
    br = np.array([-1.0, 0.0, 0.1, 0.5, -0.0, 0.2], dtype=np.float32)
    regime = np.where(br < 0, 4, np.where(br < 0.2, np.where(br == 0, 3, 2), 1)).astype(np.float32)
    want = regime * np.float32(10) + br
    out = cp.zeros(br.size, dtype=cp.float32)
    cp.RawKernel(_CROSSING, "crossing")((1,), (8,), (cp.asarray(br), out, np.int32(br.size)))
    assert cp.asnumpy(out).tobytes() == want.tobytes()


_RETRY = r'''
extern "C" __global__ void retry(const int* limit, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = 0;
    int c = 0;
label_1001:;
    if (!((k < 10) && (c == 0))) { goto label_1002; }
    k = k + 1;
    if (k >= limit[i]) { c = c + 1; }
    goto label_1001;
label_1002:;
    out[i] = k * 100 + c;
}
'''


def test_forward_goto_out_of_a_backward_retry_loop(cp):
    limit = np.array([0, 1, 3, 10, 11, 50], dtype=np.int32)
    want = []
    for lim in limit:
        k = c = 0
        while k < 10 and c == 0:
            k += 1
            c += int(k >= lim)
        want.append(k * 100 + c)
    out = cp.zeros(limit.size, dtype=cp.int32)
    cp.RawKernel(_RETRY, "retry")((1,), (8,), (cp.asarray(limit), out, np.int32(limit.size)))
    assert cp.asnumpy(out).tolist() == want


_PTX_BITS = r'''
__device__ float ptx_min(float a, float b) {
    float result;
    asm volatile("{ .reg .pred p; setp.lt.f32 p,%1,%2; selp.f32 %0,%1,%2,p; }"
                 : "=f"(result) : "f"(a), "f"(b));
    return result;
}
__device__ float ptx_neg(float a) {
    unsigned int bits;
    asm volatile("xor.b32 %0,%1,0x80000000;" : "=r"(bits) : "r"(__float_as_uint(a)));
    return __uint_as_float(bits);
}
__device__ float ptx_abs(float a) {
    float value;
    asm volatile("{ .reg .b32 bits; mov.b32 bits,%1; and.b32 bits,bits,0x7fffffff; mov.b32 %0,bits; }"
                 : "=f"(value) : "f"(a));
    return value;
}
__device__ unsigned int ptx_pick(float a, float b, unsigned int ia, unsigned int ib) {
    unsigned int chosen;
    asm volatile("{.reg .pred p; setp.gt.f32 p,%1,%2; selp.b32 %0,%3,%4,p;}"
                 : "=r"(chosen) : "f"(a), "f"(b), "r"(ia), "r"(ib));
    return chosen;
}
__device__ double ptx_dmax(double a, double b) {
    double result;
    asm volatile("{ .reg .pred p; setp.gt.f64 p,%1,%2; selp.f64 %0,%1,%2,p; }"
                 : "=d"(result) : "d"(a), "d"(b));
    return result;
}
extern "C" __global__ void ptx_bits(const float* a, const float* b, const double* da, const double* db,
                                    float* lo, float* neg, float* mag, unsigned int* pick,
                                    double* dhi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    lo[i] = ptx_min(a[i], b[i]);
    neg[i] = ptx_neg(a[i]);
    mag[i] = ptx_abs(a[i]);
    pick[i] = ptx_pick(a[i], b[i], 7u, 9u);
    dhi[i] = ptx_dmax(da[i], db[i]);
}
'''


def test_ptx_bit_operations_keep_subnormals_zeros_and_payloads(cp):
    sub = np.float32(1e-40)                         # a binary32 subnormal
    nan = np.array([0x7fc01234], dtype=np.uint32).view(np.float32)[0]
    a = np.array([sub, -sub, 0.0, -0.0, nan, 1.5, 2.0, -3.0], dtype=np.float32)
    b = np.array([0.0, 0.0, -0.0, 0.0, 1.0, nan, 2.0, sub], dtype=np.float32)
    da = np.array([5e-324, -1.0, 0.0, -0.0, np.nan, 2.0, 1.0, 3.0])
    db = np.array([0.0, -2.0, -0.0, 0.0, 1.0, np.nan, 1.0, 4.0])
    n = a.size
    outs = [cp.zeros(n, dtype=t) for t in (cp.float32, cp.float32, cp.float32, cp.uint32, cp.float64)]
    cp.RawKernel(_PTX_BITS, "ptx_bits")((1,), (8,), (cp.asarray(a), cp.asarray(b),
                                                      cp.asarray(da), cp.asarray(db), *outs, np.int32(n)))
    with np.errstate(invalid="ignore"):
        lo = np.where(a < b, a, b)                  # NumPy compares subnormals exactly
        pick = np.where(a > b, 7, 9).astype(np.uint32)
        dhi = np.where(da > db, da, db)
    neg = (a.view(np.uint32) ^ np.uint32(0x80000000)).view(np.float32)
    mag = (a.view(np.uint32) & np.uint32(0x7fffffff)).view(np.float32)
    got = [cp.asnumpy(o) for o in outs]
    assert got[0].view(np.uint32).tolist() == lo.view(np.uint32).tolist()
    assert got[1].view(np.uint32).tolist() == neg.view(np.uint32).tolist()
    assert got[2].view(np.uint32).tolist() == mag.view(np.uint32).tolist()
    assert got[3].tolist() == pick.tolist()
    assert got[4].view(np.uint64).tolist() == dhi.view(np.uint64).tolist()


_TEXT = r'''
struct Text {
    const char* text = "";
    __device__ Text() = default;
    __device__ Text(const char* s) : text(s) {}
    __device__ bool operator==(const Text& b) const {
        int i = 0;
        while (text[i] && text[i] == b.text[i]) ++i;
        return text[i] == b.text[i];
    }
};
__device__ int same(Text a, Text b) { return a == b ? 1 : 0; }
extern "C" __global__ void texts(int* out) {
    out[0] = same(Text("mynn"), Text("mynn"));
    out[1] = same(Text("mynn"), Text("ysu"));
    out[2] = same(Text(), Text(""));
}
'''


def test_string_literals_held_in_a_struct(cp):
    out = cp.zeros(3, dtype=cp.int32)
    cp.RawKernel(_TEXT, "texts")((1,), (1,), (out,))
    assert cp.asnumpy(out).tolist() == [1, 0, 1]
