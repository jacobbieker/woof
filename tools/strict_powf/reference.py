"""A correctly rounded binary32 pow, to check woof/backend/translate/
strict_powf.cuh against.  It shares no code or constants with the header.

For each (x, y), NumPy's binary64 pow is used when a rounding check shows
that its error (a few binary64 units at most; the check allows 2^-50
relative) cannot change the binary32 word.  Otherwise the value is
computed at 70 decimal digits, and a value at a binary32 midpoint (an exact
midpoint, such as 4097^2) rounds to even.  Special values are NumPy's,
which follow C99 Annex F.
"""
from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np

_CONTEXT = decimal.Context(prec=70)


def _decimal_word(ax: float, y: float) -> np.float32:
    """The correctly rounded binary32 of ax**y (ax > 0, both finite)."""
    v = _CONTEXT.exp(_CONTEXT.multiply(Decimal(y), _CONTEXT.ln(Decimal(ax))))
    f = np.float32(min(float(v), float(np.finfo(np.float32).max)))
    u = int(f.view(np.uint32))
    candidates = []
    for w in range(max(u - 2, 0), min(u + 3, 0x7F800001)):
        value = Decimal(2) ** 128 if w == 0x7F800000 else Decimal(float(np.uint32(w).view(np.float32)))
        candidates.append((abs(value - v), w))
    candidates.sort()
    (d0, w0), (d1, w1) = candidates[0], candidates[1]
    w = w0
    if d1 - d0 < v * Decimal("1e-55"):  # an exact midpoint: ties to even
        w = w0 if w0 % 2 == 0 else w1
    return np.float32(np.inf) if w >= 0x7F800000 else np.uint32(w).view(np.float32)


def powf(x, y) -> tuple[np.ndarray, int]:
    """(correctly rounded x**y as binary32, how many needed decimal)."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    with np.errstate(all="ignore"):
        d = np.power(x.astype(np.float64), y.astype(np.float64))
        lo = (d * (1 - 2.0 ** -50)).astype(np.float32)
        hi = (d * (1 + 2.0 ** -50)).astype(np.float32)
        out = d.astype(np.float32)
    ambiguous = np.flatnonzero((lo.view(np.uint32) != hi.view(np.uint32))
                               & np.isfinite(d) & (d != 0))
    for i in ambiguous:
        word = _decimal_word(abs(float(x.flat[i])), float(y.flat[i]))
        out.flat[i] = -word if d.flat[i] < 0 else word
    return out, int(ambiguous.size)
