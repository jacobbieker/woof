"""Platform-independent, correctly rounded FP64 ``exp`` and ``log``.

Not scheme-specific: any lane that derives a *pinned* constant from a
transcendental should call these rather than :mod:`math`.

Why this exists
---------------
``math.exp`` and ``math.log`` are thin wrappers over the host C library, and
the host C library is not the same program on every box.  glibc's FP64 ``exp``
(the ARM optimized-routines core, glibc >= 2.28) returns the correctly rounded
double across the ranges this project derives constants over; the Microsoft
UCRT does not.  They disagree by one ulp, and one ulp is enough to move a
SHA-256.

The concrete failure this module was written for: the aerosol-aware Thompson
droplet-number ladder evaluates ``EXP(n/100 * LOG(3000))`` for ``n`` in
1..99.  At ``n = 61`` the exact value is

    132.142940103429296889...

so the correctly rounded double is ``0x1.08492f71fb081p+7``.  glibc returns
it; the UCRT returns ``0x1.08492f71fb080p+7``, one ulp low.  That single bin
changed ``DERIVED_CONSTANT_SHA256`` and took the whole mp=28 CPU tier down on
Windows -- a pin firing on the vendor of the host libm rather than on any
change to the port.

The fix is not to loosen the pin.  A derived-constant pin exists to catch a
gamma-implementation drift, and it has to keep firing on one.  The fix is to
make the *derivation* independent of the host, so the pinned bytes are a
property of the algorithm alone.

How
---
:mod:`decimal` is a correctly rounded, software-only, fixed-precision
implementation that is part of the standard library, so it behaves identically
on every platform CPython runs on -- unlike the host libm, and unlike
``mpmath``, which is not a declared dependency of this project.
:meth:`decimal.Context.exp` and :meth:`decimal.Context.ln` are documented as
correctly rounded to the context precision.  Evaluating at 60 significant
decimal digits (~199 bits) and rounding once to binary64 (53 bits) reproduces
the correctly rounded double: the intermediate carries ~146 bits of headroom
past the rounding boundary, so the double rounding is only observable for an
argument whose exact result sits within 2**-146 relative of a tie, which does
not occur in the derivations here.  That claim is checked rather than asserted
-- ``tests/test_correctly_rounded_libm.py`` sweeps the arguments these
constants are built from against an independent arbitrary-precision reference.

These are ~12 microseconds per call, a thousand times slower than the libm
they replace.  They are for **import-time derivation of pinned constants**,
where the whole ladder costs a millisecond once.  Do not put them in a kernel
or a per-gridpoint loop; runtime physics must keep using the float32 paths in
``noahmp_libm`` and friends, which exist to reproduce a specific libm's error
rather than to avoid it.
"""

from __future__ import annotations

import decimal

__all__ = ["cbrt_cr", "exp_cr", "log_cr", "log10f_cr"]

#: 60 significant decimal digits ~= 199 bits, against the 53 a double needs.
_PRECISION = 60

_CONTEXT = decimal.Context(prec=_PRECISION)


def exp_cr(x: float) -> float:
    """``exp(x)`` correctly rounded to binary64, identically on every host."""
    return float(_CONTEXT.exp(decimal.Decimal(x)))


def log_cr(x: float) -> float:
    """``log(x)`` correctly rounded to binary64, identically on every host."""
    return float(_CONTEXT.ln(decimal.Decimal(x)))


def _icbrt(n: int) -> int:
    """``floor(n ** (1/3))`` for a non-negative integer, exactly."""
    if n < 8:
        return 1 if n else 0
    # An overestimate, then integer Newton, which descends to the floor.
    y = (1 << ((n.bit_length() + 2) // 3)) + 1
    while True:
        nxt = (2 * y + n // (y * y)) // 3
        if nxt >= y:
            break
        y = nxt
    while y * y * y > n:
        y -= 1
    while (y + 1) ** 3 <= n:
        y += 1
    return y


def cbrt_cr(x: float) -> float:
    """``cbrt(x)`` correctly rounded to binary64 (to nearest, ties to even),
    from exact integer arithmetic alone -- no libm, no decimal context.

    ``|x| = M 2^E`` exactly; with ``t`` chosen so ``cbrt(|x|) 2^t`` carries
    about 60 bits, ``A = M 2^(E + 3t)`` is an integer and ``R = floor(A^(1/3))``
    its integer cube root.  The 53-bit rounding of ``A^(1/3)`` is decided
    by comparing ``A`` with the cube of the midpoint, all in integers, so the
    result is the correctly rounded double by construction.  (A binary64
    cube root never lies exactly on a midpoint, whose cube needs more than
    53 bits; the tie branch is ties-to-even regardless.)  Slow -- tens of
    microseconds -- and meant as the oracle the fast cube roots are graded
    against (``woof.core.cr_cbrt``, ``woof/core/kernels/cr_cbrt_flt64.cuh``).
    """
    import math

    if x != x or math.isinf(x) or x == 0.0:
        return x + x
    negative = x < 0.0
    m, e = math.frexp(-x if negative else x)
    mantissa = int(m * 2.0 ** 53)          # |x| = mantissa * 2^(e - 53)
    t = 60 - e // 3
    shift = e - 53 + 3 * t                  # >= 127 for every finite x
    a = mantissa << shift
    r = _icbrt(a)
    s = r.bit_length() - 53
    q = r >> s
    mid = (2 * q + 1) << (s - 1)            # the midpoint above q, scaled
    cube = mid * mid * mid
    if a > cube or (a == cube and q & 1):
        q += 1
    value = math.ldexp(float(q), s - t)
    return -value if negative else value
def log10f_cr(values):
    """``log10`` of float32 values, correctly rounded to float32, on every host.

    The value is the 60-digit decimal logarithm rounded once to binary32:
    the nearest of the three float32 candidates around its binary64
    conversion is chosen by exact decimal distance (ties to even), so no
    double rounding can occur.  Positive finite arguments only; anything
    else is refused rather than guessed.

    This is what a C library whose ``log10f`` rounds correctly returns, and
    what P3's ``ALOG10`` evaluates to in WRF built against one.  It exists
    for P3's rain-table generation (woof/core/p3_tables.py), ~10,600
    calls once per process; the device kernel's own ``p3_log10`` is the
    run-time path.
    """
    import numpy as np

    flat = np.asarray(values, dtype=np.float32)
    out = np.empty(flat.shape, dtype=np.float32)
    src = flat.reshape(-1)
    dst = out.reshape(-1)
    for position, value in enumerate(src):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"log10f_cr takes positive finite float32, "
                             f"got {value!r}")
        exact = _CONTEXT.log10(decimal.Decimal(float(value)))
        guess = np.float32(float(exact))
        best, best_gap = None, None
        for cand in (np.nextafter(guess, np.float32(-np.inf)), guess,
                     np.nextafter(guess, np.float32(np.inf))):
            gap = abs(exact - decimal.Decimal(float(cand)))
            if (best_gap is None or gap < best_gap
                    or (gap == best_gap
                        and int(cand.view(np.uint32)) % 2 == 0)):
                best, best_gap = cand, gap
        dst[position] = best
    return out
