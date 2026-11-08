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



_COMPLEX64 = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void cz(const double* a, double* l, double* r, double* m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    thrust::complex<double> x(a[2 * i], a[2 * i + 1]);
    thrust::complex<double> lg = thrust::log(x), sq = thrust::sqrt(x);
    l[2 * i] = lg.real();
    l[2 * i + 1] = lg.imag();
    r[2 * i] = sq.real();
    r[2 * i + 1] = sq.imag();
    m[i] = thrust::abs(x);
}
"""


def test_thrust_binary64_complex_log_square_root_and_abs(cp):
    """MPAS WSM6's diagnostic radar header takes Thrust's complex<double>
    log (FreeBSD's clog: log1p near |z| = 1, the trimmed sum of squares
    between), csqrt and abs.  Every branch against NumPy's complex128."""
    a = np.array([[1.5, -2.0], [3.0, 4.0], [-3.0, 4.0], [0.6, 0.7], [0.2, 0.3],
                  [1.0, 1e-200], [1.0, 0.5], [1e-60, 2.0], [0.0, 0.0], [-0.75, -0.1]])
    n = len(a)
    l, r, m = (cp.zeros(2 * n, dtype=cp.float64), cp.zeros(2 * n, dtype=cp.float64),
               cp.zeros(n, dtype=cp.float64))
    cp.RawKernel(_COMPLEX64, "cz")((1,), (16,), (cp.asarray(a.ravel()), l, r, m, np.int32(n)))
    z = a[:, 0] + 1j * a[:, 1]
    got_l = cp.asnumpy(l).reshape(-1, 2)
    got_r = cp.asnumpy(r).reshape(-1, 2)
    with np.errstate(divide="ignore"):
        want_l = np.log(z)
    finite = np.isfinite(want_l.real)
    # clog is not correctly rounded (0.6+0.7i: one ulp from NumPy's, the trimmed sum's own word)
    np.testing.assert_allclose(got_l[finite, 0], want_l.real[finite], rtol=2e-15, atol=1e-300)
    np.testing.assert_allclose(got_l[:, 1], want_l.imag, rtol=4e-16)
    assert got_l[8, 0] == -np.inf                      # log(0): -inf + 0i
    np.testing.assert_allclose(got_r[:, 0], np.sqrt(z).real, rtol=4e-16)
    keep = np.arange(n) != 5                           # NumPy loses 1e-200i's half; csqrt keeps it
    np.testing.assert_allclose(got_r[keep, 1], np.sqrt(z).imag[keep], rtol=4e-16)
    assert got_r[5].tolist() == [1.0, 5e-201]
    np.testing.assert_allclose(cp.asnumpy(m), np.abs(z), rtol=4e-16)
    assert got_r[1].tolist() == [2.0, 1.0] and got_r[2].tolist() == [1.0, 2.0]


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


def test_a_second_jump_from_inside_the_retry_loop_is_dispatched(cp):
    """The entry guard skips the loop's head only on the first pass: a jump
    to the same label from inside the guarded statements cannot be spelled
    that way (its break would leave the retry loop).  It was refused; the
    labels are now a dispatch loop (a switch on the label), and the kernel
    gives the words of the Fortran's control flow."""
    from woof.backend.translate import translate
    assert "goto label_100; }\n    visits" in _ENTRY_FROM_INSIDE
    assert "switch (__woof_st" in translate(_ENTRY_FROM_INSIDE, unit="entry_from_inside").msl
    # the same shape, bounded: the jump back stops once r reaches 30
    src = _ENTRY_FROM_INSIDE.replace("if ((r > 5.0f))", "if ((r > 5.0f) && (r < 30.0f))")
    assert src != _ENTRY_FROM_INSIDE
    q = np.array([0.0, 2.0, 1.5, -3.0, 5.0, -9.0], dtype=np.float32)
    qs = np.array([1.0, 1.0, 2.0, 0.0, 5.0, 9.0], dtype=np.float32)
    want = []
    for q1, qs1 in zip(q, qs):
        r, visits = np.float32(0), 0
        label = "100" if q1 < qs1 else "90"
        while label != "200":
            if label == "90":
                r += np.float32(1)
                if np.float32(5) < r < np.float32(30):
                    label = "100"
                    continue
                visits += 1
                label = "200"
            else:
                r += np.float32(10)
                q1 = q1 + np.float32(1)
                label = "90" if q1 > qs1 else "200"
        want.append(r * np.float32(100) + np.float32(visits))
    want = np.array(want, dtype=np.float32)
    out = cp.zeros(q.size, dtype=cp.float32)
    cp.RawKernel(src, "entry")((1,), (8,), (cp.asarray(q), cp.asarray(qs), out, np.int32(q.size)))
    assert cp.asnumpy(out).tobytes() == want.tobytes()


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
    monkeypatch.delenv("WOOF_HEX_METAL", raising=False)     # the default on mlx
    assert metal_lane.install()
    options = hex_numerics.FAST_KERNEL_BASE
    assert not loader.contracts(options, _CUDA_SOURCE)
    assert "fp contract(fast)" not in loader.translate_unit("prep", _CUDA_SOURCE, options=options).msl


_WSM6_SHAPES = r'''
namespace detail {
template <class A> struct is_arithmetic { static constexpr bool value = false; };
template <> struct is_arithmetic<int> { static constexpr bool value = true; };
template <> struct is_arithmetic<float> { static constexpr bool value = true; };
template <bool, class T = int> struct enable_if {};
template <class T> struct enable_if<true, T> { using type = T; };
}
template <class B, typename detail::enable_if<detail::is_arithmetic<B>::value, int>::type = 0>
__device__ float twice(B b) { return 2.0f * (float)b; }
struct State { const char* last_message; int flag; };
__device__ void init(State& s) { s.last_message = "mp_init OK"; s.flag = 0; }
__device__ int length(const char* text) { int n = 0; while (text[n]) ++n; return n; }
struct Rows { float* p; long long n; __device__ float& operator[](int k) const { return p[k * n]; } };
template <class A> __device__ float settle(A rows, int s) {
    float v = static_cast<volatile float&>(rows[s + 1]);
    return v + __expf(0.0f);
}
extern "C" __global__ void shapes(float* work, int* out) {
    State s;
    init(s);
    out[0] = length(s.last_message);
    out[1] = (int)twice(3) + (int)twice(1.5f);
    const Rows rows{work, 1};
    work[0] = settle(rows, 0);
}
'''


def test_wsm6_radar_unit_shapes(cp):
    """Constructs MPAS WSM6's radar unit and RRFS's convective diagnostics
    brought in: a type trait's explicit specializations (their static
    members stay folded, as a class template's do), a state field given a
    string literal by a free function and then passed on (constant data),
    a volatile load through a by-value accessor (``__woof_vload`` finds the
    space), and CUDA's ``__expf``."""
    work = cp.asarray(np.array([0.0, 2.5], dtype=np.float32))
    out = cp.zeros(2, dtype=cp.int32)
    cp.RawKernel(_WSM6_SHAPES, "shapes")((1,), (1,), (work, out))
    assert cp.asnumpy(out).tolist() == [len("mp_init OK"), 9]
    assert float(cp.asnumpy(work)[0]) == 3.5          # 2.5 + __expf(0) == 2.5 + 1


_LOOP_AT_LABEL = r'''
extern "C" __global__ void loop_at_label(const int* q, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = 0;
    while (k < 5) { k = k + 1; if (k == q[i]) goto L; }L: out[i] = k * 10 + (k == q[i] ? 1 : 0);
}
'''


def test_a_loop_that_ends_at_its_jumps_label(cp):
    """The loop's ``if (flag) break;`` and the wrapper's close land at one
    offset: the break goes first, inside the wrapper."""
    q = np.array([2, 7, 5], dtype=np.int32)
    out = cp.zeros(3, dtype=cp.int32)
    cp.RawKernel(_LOOP_AT_LABEL, "loop_at_label")((1,), (4,), (cp.asarray(q), out, np.int32(3)))
    assert cp.asnumpy(out).tolist() == [21, 50, 51]


def test_the_hex_lane_refuses_a_paravirtual_device(cp, monkeypatch):
    """A virtual Mac's "Apple Paravirtual device" computes NaN where an
    M-series GPU matches: the hex lane is the default on an Apple GPU only."""
    from woof.hex import metal_lane
    assert metal_lane.apple_gpu()
    props = dict(cp.cuda.runtime.getDeviceProperties(0), architecture="Apple Paravirtual device")
    monkeypatch.setattr(cp.cuda.runtime, "getDeviceProperties", lambda device: props)
    assert not metal_lane.apple_gpu()
