"""Gradient checks on hand-written TL/AD kernels (arrays, not autograd).

The argument shapes follow woof.grad.checks (P1): sequences of arrays,
inner products and norms in binary64 on the host, so the two kits can be
unified later.  ``tl``/``ad``/``model`` callables take and return
sequences of CuPy (or facade) or NumPy arrays.
"""

from __future__ import annotations

import math

import numpy as np


def _host(a):
    if isinstance(a, np.ndarray):
        return a
    import cupy as cp
    return cp.asnumpy(a)


def inner(xs, ys) -> float:
    """<x, y> over a sequence of arrays, in binary64, in sequence order."""
    total = 0.0
    for x, y in zip(xs, ys, strict=True):
        total += float(np.dot(_host(x).astype(np.float64).ravel(),
                              _host(y).astype(np.float64).ravel()))
    return total


def norm(xs) -> float:
    return math.sqrt(inner(xs, xs))


def dot_product(tl, ad, dx, dy, *, terms: bool = False) -> float:
    """Relative misfit of <TL dx, dy> against <dx, AD dy>.

    ``terms=True`` scales by the inner products of the magnitudes
    (<|TL dx|, |dy|> and <|dx|, |AD dy|>) instead of the results: the
    roundoff bound when the terms cancel, as they can on small grids in
    binary32.
    """
    tdx, ady = tl(dx), ad(dy)
    left = inner(tdx, dy)
    right = inner(dx, ady)
    scale = max(abs(left), abs(right), np.finfo(np.float64).tiny)
    if terms:
        mag = lambda xs: [np.abs(_host(x)) for x in xs]
        scale = max(scale, inner(mag(tdx), mag(dy)), inner(mag(dx), mag(ady)))
    return abs(left - right) / scale


def tl_convergence(model, tl, x, dx, alphas) -> list[tuple[float, float]]:
    """``||M(x + a dx) - M(x)|| / ||a TL dx||`` for each ``a``; -> 1."""
    base = [_host(y).astype(np.float64) for y in model(x)]
    lin = [_host(y).astype(np.float64) for y in tl(dx)]
    lin_norm = math.sqrt(sum(float(np.vdot(y, y)) for y in lin))
    out = []
    for a in alphas:
        xp = [xi + a * di for xi, di in zip(x, dx, strict=True)]
        moved = [_host(y).astype(np.float64) for y in model(xp)]
        diff = math.sqrt(sum(float(np.vdot(m - b, m - b))
                             for m, b in zip(moved, base, strict=True)))
        out.append((float(a), diff / (abs(a) * lin_norm)))
    return out


def taylor(cost, grad, x, h, alphas) -> list[tuple[float, float]]:
    """``(J(x + a h) - J(x)) / (a <grad J(x), h>)`` for each ``a``; -> 1."""
    j0 = float(cost(x))
    slope = inner(grad(x), h)
    out = []
    for a in alphas:
        xp = [xi + a * hi for xi, hi in zip(x, h, strict=True)]
        out.append((float(a), (float(cost(xp)) - j0) / (a * slope)))
    return out
