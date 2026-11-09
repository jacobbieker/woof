"""f32s (binary32 with gradual underflow on Metal) against IEEE binary32.

In an ``--ftz=false`` unit the translator spells ``float`` as ``f32s``
(woof/backend/metal/f32s.h): the hardware operation where no flush can have
happened, soft binary64 otherwise.  An exact zero sum -- ``0 + 0``, ``x - x``,
``0 * y + 0`` -- is such a case and takes the hardware path; before, every
one took soft binary64, and a stencil over a field of zeros (an absent
hydrometeor) ran 12x slower than plain float.  The host's NumPy binary32
keeps subnormals and is correctly rounded, so it is the reference for
+ - * /, signed zeros included.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_OPS = r'''
extern "C" __global__ void ops(const float* a, const float* b, const float* c, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], z = c[i];
    o[5 * i + 0] = x + y;
    o[5 * i + 1] = x - y;
    o[5 * i + 2] = x * y;
    o[5 * i + 3] = x / y;
    o[5 * i + 4] = fmaf(x, y, z);
}
'''


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


def _operands():
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
                     0x00800001, 0x00FFFFFF, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
                     0x7F800000, 0xFF800000, 0x1F800000, 0x9F800000, 0x00400000], dtype=np.uint32)
    special = np.concatenate([bits.view(np.float32), np.float32([1.5, -2.25, 3e-38, -3e-38, 0.1, -0.7])])
    a, b = (v.ravel() for v in np.meshgrid(special, special, indexing="ij"))
    rng = np.random.default_rng(7)
    m = 1 << 16
    raw = rng.integers(0, 1 << 32, m, dtype=np.uint64).astype(np.uint32)
    expo = rng.choice([0, 1, 2, 3, 20, 127], m).astype(np.uint32)
    raw = (raw & 0x807FFFFF) | (expo << 23)
    raw[rng.random(m) < 0.2] &= 0x80000000                  # signed zeros
    r = raw.view(np.float32)
    s = np.roll(r, 1)
    s = np.where(rng.random(m) < 0.25, -r, s)               # x + (-x)
    s = np.where(rng.random(m) < 0.15, r, s)                # x - x
    return np.concatenate([a, r]), np.concatenate([b, s])


def _same_words(got, want):
    nan = np.isnan(got) & np.isnan(want)
    return nan | (got.view(np.uint32) == want.view(np.uint32))


def test_f32s_arithmetic_is_ieee_binary32_with_signed_zeros(cp):
    a, b = _operands()
    zeros = np.zeros_like(a)
    n = a.size
    fn = cp.RawModule(code=_OPS, options=("--ftz=false",)).get_function("ops")
    assert "f32s" in fn.unit.msl
    o = cp.zeros(5 * n, dtype=cp.float32)
    fn(((n + 255) // 256,), (256,), (cp.asarray(a), cp.asarray(b), cp.asarray(zeros), o, np.int32(n)))
    got = o.get().reshape(n, 5)
    with np.errstate(all="ignore"):
        want = {0: a + b, 1: a - b, 2: a * b, 3: a / b}
    for column, reference in want.items():
        ok = _same_words(got[:, column], reference.astype(np.float32))
        assert ok.all(), (column, a[~ok][:4], b[~ok][:4], got[~ok, column][:4], reference[~ok][:4])
    # fmaf(x, y, 0): a zero product plus +0 is the IEEE zero of the sum
    product_zero = (a == 0) | (b == 0)
    finite = np.isfinite(a) & np.isfinite(b)
    pick = product_zero & finite
    with np.errstate(all="ignore"):
        want_fma = (a * b + np.float32(0.0)).astype(np.float32)
    assert _same_words(got[pick, 4], want_fma[pick]).all()


def test_exact_zero_sums_keep_ieee_signs(cp):
    pz, nz = np.float32(0.0), np.float32(-0.0)
    one = np.float32(1.0)
    tiny = np.uint32(0x00800000).view(np.float32)            # smallest normal
    a = np.float32([pz, nz, pz, nz, one, tiny, -tiny, tiny])
    b = np.float32([pz, nz, nz, pz, -one, -tiny, tiny, np.float32(-1.5) * tiny])
    c = np.float32([pz, nz, nz, pz, pz, nz, pz, pz])
    n = a.size
    fn = cp.RawModule(code=_OPS, options=("--ftz=false",)).get_function("ops")
    o = cp.zeros(5 * n, dtype=cp.float32)
    fn((1,), (32,), (cp.asarray(a), cp.asarray(b), cp.asarray(c), o, np.int32(n)))
    got = o.get().reshape(n, 5).view(np.uint32)
    assert np.array_equal(got[:, 0], (a + b).view(np.uint32))
    assert np.array_equal(got[:, 1], (a - b).view(np.uint32))
    # tiny + (-1.5 tiny) = -0.5 tiny: a subnormal, kept, not flushed to zero
    assert got[7, 0] == (a[7] + b[7]).view(np.uint32) != 0x80000000
    # 0 * y + z with zero z: signed zero by IEEE's sum rule
    assert np.array_equal(got[:4, 4], (a[:4] * b[:4] + c[:4]).view(np.uint32))
