"""Correctly rounded binary64 transcendentals for the WRFDA host-side ports.

WRFDA's observation preprocessing (map projection, wind rotation, the
saturation humidity behind ``q`` from ``rh``) calls ``sin``, ``cos``, ``tan``,
``exp``, ``log10`` and real ``**``. On a glibc host :mod:`math` gives the
oracle's own words. Apple's libm and the UCRT disagree with glibc by an ulp
or two on about a fifth of the tutorial's observation positions, and an ulp
in ``dx`` reaches the binary32 production tier's rounding. That would break
the CUDA/Metal identity, because the two backends' hosts would preprocess
differently.

These functions evaluate in :mod:`decimal` at 80 significant digits (about
265 bits) and round once to binary64, so the result belongs to the algorithm
and not to the host. :mod:`woof.core.correctly_rounded_libm` makes the same
argument for ``exp`` and ``log``. glibc's result is the correctly rounded one
on all but a few of the tutorial's arguments; ``tests/test_var_obs_wrfda.py``
records those.

They are scalar, host-side and slow (tens of microseconds per call). Use them
for per-observation preprocessing, never in a kernel or a per-gridpoint loop.
"""

from __future__ import annotations

import decimal
import math

__all__ = ["sin", "cos", "tan", "exp", "log10", "pow_", "sqrt"]

_C = decimal.Context(prec=80, rounding=decimal.ROUND_HALF_EVEN, Emax=999999, Emin=-999999)
_TINY = decimal.Decimal("1e-85")


def _pi() -> decimal.Decimal:
    # Machin's formula at the working precision.
    def arctan_inv(n: int) -> decimal.Decimal:
        x = _C.divide(1, n)
        n2 = n * n
        term, total, k, sign = x, x, 1, -1
        while True:
            term = _C.divide(term, n2)
            add = _C.divide(term, 2 * k + 1)
            if add < _TINY:
                return total
            total = _C.add(total, add) if sign > 0 else _C.subtract(total, add)
            sign, k = -sign, k + 1

    return _C.subtract(_C.multiply(16, arctan_inv(5)), _C.multiply(4, arctan_inv(239)))


_PI = _pi()
_TWO_PI = _C.multiply(2, _PI)


def _sin_cos(x: float) -> tuple[decimal.Decimal, decimal.Decimal]:
    r = decimal.Decimal(x)
    if not -_PI < r <= _PI:
        # The arguments here are a few radians at most; 80 digits of pi
        # leave far more than 53 bits after this subtraction.
        r = _C.subtract(r, _C.multiply(_C.to_integral_value(_C.divide(r, _TWO_PI)), _TWO_PI))
    r2 = _C.multiply(r, r)
    s = term = r
    k = 1
    while abs(term) >= _TINY:
        term = _C.divide(_C.multiply(-term, r2), (2 * k) * (2 * k + 1))
        s = _C.add(s, term)
        k += 1
    c = term = decimal.Decimal(1)
    k = 1
    while abs(term) >= _TINY:
        term = _C.divide(_C.multiply(-term, r2), (2 * k - 1) * (2 * k))
        c = _C.add(c, term)
        k += 1
    return s, c


def sin(x: float) -> float:
    return x if x == 0.0 else float(_sin_cos(x)[0])


def cos(x: float) -> float:
    return float(_sin_cos(x)[1])


def tan(x: float) -> float:
    if x == 0.0:
        return x
    s, c = _sin_cos(x)
    return float(_C.divide(s, c))


def exp(x: float) -> float:
    return float(_C.exp(decimal.Decimal(x)))


def log10(x: float) -> float:
    if not x > 0.0:
        raise ValueError(f"log10 of {x!r}")
    return float(_C.log10(decimal.Decimal(x)))


def pow_(x: float, y: float) -> float:
    """Fortran ``x ** y`` with a real exponent, for ``x > 0``."""
    if not x > 0.0:
        raise ValueError(f"pow of {x!r}")
    if y == 0.0:
        return 1.0
    return float(_C.exp(_C.multiply(decimal.Decimal(y), _C.ln(decimal.Decimal(x)))))


def sqrt(x: float) -> float:
    # IEEE sqrt is correctly rounded on every host.
    return math.sqrt(x)
