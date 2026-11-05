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
* ``#include <cupy/complex.cuh>`` (MPAS WSM6's radar arithmetic): a Mac has
  no CuPy headers; the translator's stand-in is Thrust's text.
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


_COMPLEX = r'''
#include <cupy/complex.cuh>
extern "C" __global__ void cdiv(const float* a, const float* b, float* q, float* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    thrust::complex<float> x(a[2 * i], a[2 * i + 1]), y(b[2 * i], b[2 * i + 1]);
    thrust::complex<float> z = x / y;
    q[2 * i] = z.real();
    q[2 * i + 1] = z.imag();
    thrust::complex<float> s = thrust::sqrt(x);
    r[2 * i] = s.real();
    r[2 * i + 1] = s.imag();
}
'''


def test_thrust_complex_division_and_square_root(cp):
    """MPAS WSM6's radar arithmetic includes <cupy/complex.cuh>; a Mac has no
    CuPy headers, so the translator's stand-in carries Thrust's division
    (scaled by |re| + |im|) and csqrtf, operation for operation."""
    a = np.array([[1.5, -2.0], [3.0, 4.0], [-3.0, 4.0], [0.0, 0.0], [1e-3, 7.25]], dtype=np.float32)
    b = np.array([[0.5, 0.25], [2.0, -1.0], [1.0, 0.0], [3.0, 4.0], [-6.0, 8.0]], dtype=np.float32)
    f = np.float32
    want_q = []
    for (ar, ai), (br, bi) in zip(a, b):
        s = abs(br) + abs(bi)
        oos = f(1) / s
        ars, ais, brs, bis = ar * oos, ai * oos, br * oos, bi * oos
        s = (brs * brs) + (bis * bis)
        oos = f(1) / s
        want_q += [((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos]
    want_q = np.array(want_q, dtype=np.float32)
    q = cp.zeros(a.size, dtype=cp.float32)
    r = cp.zeros(a.size, dtype=cp.float32)
    cp.RawKernel(_COMPLEX, "cdiv")((1,), (8,), (cp.asarray(a.ravel()), cp.asarray(b.ravel()), q, r,
                                                 np.int32(len(a))))
    assert cp.asnumpy(q).view(np.uint32).tolist() == want_q.view(np.uint32).tolist()
    # sqrt(3+4i) = 2+i and sqrt(-3+4i) = 1+2i: hypot(3, 4) = 5 is exact in any libm
    got_r = cp.asnumpy(r).reshape(-1, 2)
    assert got_r[1].tolist() == [2.0, 1.0] and got_r[2].tolist() == [1.0, 2.0]
    assert got_r[3].tolist() == [0.0, 0.0]


_ENTRY = r'''
extern "C" __global__ void entry(const float* q, const float* qs, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float q1 = q[i], qs1 = qs[i];
    float r = 0.0f;
    int visits = 0;
    if ((q1 < qs1)) { goto label_100; }
    label_90: ;
    r = r + 1.0f;
    visits = visits + 1;
    goto label_200;
    label_100: ;
    r = r + 10.0f;
    q1 = q1 + 1.0f;
    if ((q1 > qs1)) { goto label_90; }
    label_200: ;
    out[i] = r * 100.0f + (float)visits;
}
'''


def test_forward_goto_into_a_retry_loop(cp):
    """RUC LSM's shape: a jump from before a retry loop to a label inside it;
    the loop's statements before that label are skipped on the first pass."""
    q = np.array([0.0, 2.0, 1.5, -3.0, 5.0], dtype=np.float32)
    qs = np.array([1.0, 1.0, 2.0, 0.0, 5.0], dtype=np.float32)
    want = []
    for q1, qs1 in zip(q, qs):
        r, visits = np.float32(0), 0
        if q1 < qs1:
            r += np.float32(10)
            q1 = q1 + np.float32(1)
            if q1 > qs1:
                r += np.float32(1)
                visits += 1
        else:
            r += np.float32(1)
            visits += 1
        want.append(r * np.float32(100) + np.float32(visits))
    want = np.array(want, dtype=np.float32)
    out = cp.zeros(q.size, dtype=cp.float32)
    cp.RawKernel(_ENTRY, "entry")((1,), (8,), (cp.asarray(q), cp.asarray(qs), out, np.int32(q.size)))
    assert cp.asnumpy(out).tobytes() == want.tobytes()


_ENTRY_FROM_INSIDE = _ENTRY.replace(
    "    r = r + 1.0f;\n    visits = visits + 1;\n",
    "    r = r + 1.0f;\n    if ((r > 5.0f)) { goto label_100; }\n    visits = visits + 1;\n")


def test_a_second_jump_from_inside_the_retry_loop_is_refused():
    """The entry guard skips the loop's head only on the first pass: a jump
    to the same label from inside the guarded statements cannot be spelled
    that way (its break would leave the retry loop), so it is refused."""
    from woof.backend.translate import TranslationError, translate
    assert "goto label_100; }\n    visits" in _ENTRY_FROM_INSIDE
    with pytest.raises(TranslationError, match="goto"):
        translate(_ENTRY_FROM_INSIDE, unit="entry_from_inside")


_CHECKED = r'''
extern "C" __global__ void produce(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i] + b[i] * a[i + 1];
}
'''


@pytest.fixture
def fast(monkeypatch):
    """Fast numerics for this test (the battery runs strict)."""
    from woof import numerics
    monkeypatch.setattr(numerics, "_mode", "fast")


def test_a_registered_source_never_contracts(fast):
    """A unit whose kernels compare each other's words translates without
    contraction in fast numerics too; any other unit still contracts."""
    from woof.backend.metal import loader
    fast = ("--std=c++17", "--fmad=true")
    other = _CHECKED.replace("produce", "produce_other")
    assert loader.contracts(fast, _CHECKED) and loader.contracts(fast, other)
    assert "fp contract(fast)" in loader.translate_unit("checked", _CHECKED, options=fast).msl
    loader.uncontracted_source(_CHECKED)
    assert not loader.contracts(fast, _CHECKED)
    assert loader.contracts(fast, other)
    assert "fp contract(fast)" not in loader.translate_unit("checked", _CHECKED, options=fast).msl


def test_the_hex_lane_prepares_mpas_to_physics_uncontracted(cp, fast, monkeypatch):
    """validate_prep_v841_f32 recomputes the preparation's words and compares
    them.  Metal's contract(fast) fused the producer and the checker
    differently on a real HRRR start, and every fast run refused at step 0."""
    from woof.backend.metal import loader
    from woof.hex import metal_lane
    from woof.hex import numerics as hex_numerics
    from woof.hex.cuda_physics_prep_v841 import _CUDA_SOURCE
    monkeypatch.setenv("WOOF_HEX_METAL", "1")
    assert metal_lane.install()
    options = hex_numerics.FAST_KERNEL_BASE
    assert not loader.contracts(options, _CUDA_SOURCE)
    assert "fp contract(fast)" not in loader.translate_unit("prep", _CUDA_SOURCE, options=options).msl
