"""THIRD-PARTY NOTICE: a NumPy transcription of CORE-MATH's binary64 cbrt.

CORE-MATH ``src/binary64/cbrt/cbrt.c`` (revision
8ae55cdd0d953dc1d6f78659435283d3436865ab, kept verbatim and sha256-pinned at
``tools/uwpbl_wrf471_oracle/libm64/upstream/cbrt.c``):
"Copyright (c) 2021-2022 Alexei Sibidanov.", MIT; the permission notice is
reproduced in ``woof/core/kernels/cr_cbrt_flt64.cuh`` and at
``licenses/LICENSE-CORE-MATH-binary64-MIT.txt``.  This adaptation is
distributed under Apache-2.0.

A correctly rounded (round to nearest, ties to even) binary64 cube root,
the host twin of the device ``uw_cbrt`` in
``woof/core/kernels/cr_cbrt_flt64.cuh``.  Both are the same algorithm on
IEEE binary64 operations, and both are correctly rounded, so they agree to
the last bit -- and agree with any other correctly rounded cbrt, which is
what lets a cube root be shared between a CUDA kernel and a host product:
``np.cbrt`` is the C library's, and glibc's is not correctly rounded
(measured: up to 3 ULP from ``pow(x, 1/3)`` and not always nearest).

THE ONLY ARITHMETIC CHANGES FROM UPSTREAM.  Upstream's ``__builtin_fma``
appears only as ``fma(a, b, -fl(a*b))`` -- the exact rounding error of a
product -- and NumPy has no fused multiply-add, so each is Dekker's
error-free product (Veltkamp split), which returns that same exactly
representable value for the operands here (|a|, |b| in [0.6, 4]: no
overflow or underflow).  Upstream normalises a subnormal argument with
``clz``; here the subnormal is scaled by 2^54 first and the result by
2^-18, both exact.  Rounding to nearest only, as on the device.

PROOF: ``tools/uwpbl_wrf471_oracle/libm64/README.md`` ("cbrt"): equal to
the upstream C, to the device header and to the exact integer reference
:func:`woof.core.correctly_rounded_libm.cbrt_cr`;
``tests/test_cr_cbrt.py`` re-checks a sweep on every run.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEVICE_HEADER", "MODULE_OPTIONS", "cbrt", "device_cbrt",
           "device_header_source"]

_U64 = np.uint64
#: Upstream writes every constant as a C99 hexadecimal float; so does this.
_hex = float.fromhex
_SPLIT = 134217729.0          # 2^27 + 1, Veltkamp's splitter
_ESCALE = np.array([1.0, _hex("0x1.428a2f98d728bp+0"), _hex("0x1.965fea53d6e3dp+0")])
_C0, _C1 = _hex("0x1.1b0babccfef9cp-1"), _hex("0x1.2c9a3e94d1da5p-1")
_C2, _C3 = -_hex("0x1.4dc30b1a1ddbap-3"), _hex("0x1.7a8d3e4ec9b07p-6")
_UU0, _UU1 = _hex("0x1.5555555555555p-2"), _hex("0x1.c71c71c71c71cp-3")
_RSC = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
_OFF = _hex("0x1p-53")
#: Subnormal scaling: 2^54 in, 2^-18 (= 2^(-54/3)) out, both exact.
_TINY, _UP, _DOWN = _hex("0x1p-1022"), _hex("0x1p54"), _hex("0x1p-18")


def _product_error(a, b, p):
    """``fma(a, b, -p)`` for ``p = fl(a*b)``: Dekker's exact error term."""
    c = _SPLIT * a
    ah = c - (c - a)
    al = a - ah
    c = _SPLIT * b
    bh = c - (c - b)
    bl = b - bh
    return (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def _newton(y, zz, rr):
    """Upstream's linear Newton step on y: returns (y1, dy, ady0, ady1)."""
    y2 = y * y
    y2l = _product_error(y, y, y2)
    y3 = y2 * y
    y3l = _product_error(y, y2, y3) + y * y2l
    h = ((y3 - zz) + y3l) * rr
    dy = h * (y * _UU0)
    y1 = y - dy
    dy = (y - y1) - dy
    ady = np.abs(dy)
    return (y1, dy, np.abs(ady - _OFF), np.abs(ady - (_hex("0x1p-52") + _OFF)))


def _normal(x):
    """CORE-MATH cr_cbrt on finite, nonzero, NORMAL binary64 ``x``."""
    hx = x.view(_U64)
    mant = hx & (~_U64(0) >> _U64(12))
    sign = hx >> _U64(63)
    e = ((hx >> _U64(52)) & _U64(0x7ff)) + _U64(3072)
    cvt1 = mant | (_U64(0x3ff) << _U64(52))
    et = e // _U64(3)
    it = e % _U64(3)
    cvt5 = (cvt1 + (it << _U64(52))) | (sign << _U64(63))
    zz = cvt5.view(np.float64)
    isc = _ESCALE[it.astype(np.intp)].view(_U64) | (sign << _U64(63))
    esc = isc.view(np.float64)
    z = cvt1.view(np.float64)
    r = 1.0 / z
    rr = r * _RSC[((it << _U64(1)) | sign).astype(np.intp)]
    z2 = z * z
    c0 = _C0 + z * _C1
    c2 = _C2 + z * _C3
    y = c0 + z2 * c2
    y2 = y * y
    h = y2 * (y * r) - 1.0
    y = y - (h * y) * (_UU0 - _UU1 * h)
    y = y * esc
    y1, dy, ady0, ady1 = _newton(y, zz, rr)
    hard = (ady0 < _hex("0x1p-75")) | (ady1 < _hex("0x1p-75"))
    if hard.any():
        y1h, dyh, a0, a1 = _newton(y1[hard], zz[hard], rr[hard])
        harder = (a0 < _hex("0x1p-98")) | (a1 < _hex("0x1p-98"))
        if harder.any():
            azz = np.abs(zz[hard])
            for arg, root in ((_hex("0x1.9b78223aa307cp+1"), _hex("0x1.79d15d0e8d59cp+0")),
                              (_hex("0x1.a202bfc89ddffp+2"), _hex("0x1.de87aa837820fp+0"))):
                hit = harder & (azz == arg)
                y1h = np.where(hit, np.copysign(root, zz[hard]), y1h)
        y1[hard] = y1h
        dy[hard] = dyh
    cvt3 = y1.view(_U64) + ((et.astype(np.int64) - 342 - 1023)
                            .astype(_U64) << _U64(52))
    m0 = (cvt3 << _U64(30)).view(np.int64)
    m1 = m0 >> np.int64(63)
    exact = (m0 ^ m1).view(_U64) <= (_U64(1) << _U64(30))
    if exact.any():
        cvt4 = ((y1.view(_U64) + (_U64(1) << _U64(15)))
                & _U64(0xffffffffffff0000))
        fix = exact & ((np.abs((cvt4.view(np.float64) - y1) - dy) < _hex("0x1p-60"))
                       | (np.abs(zz) == 1.0))
        cvt3 = np.where(
            fix, (cvt3 + (_U64(1) << _U64(15))) & _U64(0xffffffffffff0000),
            cvt3)
    return cvt3.view(np.float64)


#: Arguments per pass: ``_normal`` holds about fifteen temporaries of its
#: input's size, so a forecast-sized array is taken a block at a time.
_BLOCK = 1 << 18


def _block(x: np.ndarray) -> np.ndarray:
    """One block: the common all-normal case straight through, else masked."""
    a = np.abs(x)
    if np.all((a >= _TINY) & (a <= np.finfo(np.float64).max)):
        return _normal(x)
    finite = np.isfinite(x) & (x != 0.0)
    # 0, inf, nan: upstream's ``x + x`` (a signalling NaN quietens).
    with np.errstate(invalid="ignore"):
        out = np.where(finite, 0.0, x + np.where(finite, 0.0, x))
    if finite.any():
        v = x[finite]
        subnormal = np.abs(v) < _TINY
        v[subnormal] = v[subnormal] * _UP
        root = _normal(v)
        root[subnormal] = root[subnormal] * _DOWN
        out[finite] = root
    return out


def cbrt(values) -> np.ndarray:
    """Correctly rounded binary64 cube root, element by element."""
    x = np.ascontiguousarray(values, dtype=np.float64)
    shape = x.shape
    flat = x.reshape(-1)
    if flat.size <= _BLOCK:
        return _block(flat).reshape(shape)
    out = np.empty_like(flat)
    for lo in range(0, flat.size, _BLOCK):
        out[lo:lo + _BLOCK] = _block(flat[lo:lo + _BLOCK])
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# The device twin, batched (the proof sweeps and the parity tests call it)
# ---------------------------------------------------------------------------

#: The device header, and the options every module that includes it uses.
DEVICE_HEADER = "cr_cbrt_flt64.cuh"
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.cr_cbrt:cbrt_batch"

_BATCH_SOURCE = r"""
extern "C" __global__ void cr_cbrt_batch(const double* __restrict__ x,
                                         double* __restrict__ y,
                                         const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = uw_cbrt(x[i]);
}
"""


def device_header_source() -> str:
    """The device header's text, read from the kernel directory."""
    from pathlib import Path

    return (Path(__file__).parent / "kernels" / DEVICE_HEADER).read_text(
        encoding="utf-8")


def _batch_module(device_id: int):
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    cached = _BATCH_MODULES.get(device_id)
    if cached is not None:
        return cached
    source = device_header_source() + _BATCH_SOURCE
    with cp.cuda.Device(device_id):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None,
                                              "cr_cbrt_batch.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    _BATCH_MODULES[device_id] = module
    return module


_BATCH_MODULES: dict = {}


def device_cbrt(values):
    """``uw_cbrt`` over a CuPy float64 array (device in, device out)."""
    import cupy as cp

    x = cp.ascontiguousarray(cp.asarray(values, dtype=cp.float64))
    y = cp.empty_like(x)
    n = int(x.size)
    if n:
        kernel = _batch_module(int(x.device.id)).get_function("cr_cbrt_batch")
        kernel(((n + 255) // 256,), (256,), (x, y, np.int64(n)))
    return y
