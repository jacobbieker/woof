"""Control-vector inner products and updates with one summation order
(docs/dev/WRFDA-3DVAR-PLAN.md, section 4.7).

Two reduction modes, both accumulating in binary64:

``wrfda-serial``
    ``da_dot``'s order: one left-to-right sum of the products, added by one
    device thread (a block forms the products in parallel, tile by tile).
    The oracle tier's mode; slow, for comparison and tests.
``fixed-tree``
    chunks of 4,096 values, each summed by one block (16 consecutive values
    per thread in index order, then a fixed binary tree over 256 threads),
    and the chunk partials combined the same way by one block.  No atomics
    and no vendor reduction, so the order is a function of the length
    alone: CUDA and Metal give the same words under strict numerics.
"""

from __future__ import annotations

import numpy as np

from woof.da.var.transforms import PRECISIONS, blocks, kernel

__all__ = ["REDUCTIONS", "dot", "dot_device", "axpy", "axpy_neg_device"]

REDUCTIONS = ("wrfda-serial", "fixed-tree")
_CHUNK = 256 * 16


def _precision(x) -> str:
    if x.dtype == np.float64:
        return PRECISIONS[0]
    if x.dtype == np.float32:
        return PRECISIONS[1]
    raise TypeError(f"control vectors are float64 or float32, not {x.dtype}")


def dot(x, y, reduction: str = "fixed-tree") -> float:
    """``<x, y>`` in binary64 with ``reduction``'s fixed order."""
    return float(dot_device(x, y, reduction).get()[0])


def dot_device(x, y, reduction: str = "fixed-tree"):
    """:func:`dot` left on the device: a one-value binary64 array (no host sync)."""
    import cupy as cp

    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, not {reduction!r}")
    if x.dtype != y.dtype or x.size != y.size:
        raise ValueError(f"dot of {x.dtype}[{x.size}] and {y.dtype}[{y.size}]")
    prec = _precision(x)
    x = cp.ascontiguousarray(x).ravel()
    y = cp.ascontiguousarray(y).ravel()
    n = np.int64(x.size)
    out = cp.zeros(1, dtype=cp.float64)
    if reduction == "wrfda-serial":
        kernel(prec, "var_dot_serial")((1,), (256,), (x, y, n, out))
    else:
        count = max(1, (int(n) + _CHUNK - 1) // _CHUNK)
        partial = cp.zeros(count, dtype=cp.float64)
        kernel(prec, "var_dot_chunks")((count,), (256,), (x, y, n, partial))
        kernel(prec, "var_dot_combine")((1,), (256,), (partial, np.int64(count), out))
    return out


def axpy(a: float, x, y) -> None:
    """``y = y + a x`` in place, one rounding per operation."""
    if x.dtype != y.dtype or x.size != y.size:
        raise ValueError(f"axpy of {x.dtype}[{x.size}] into {y.dtype}[{y.size}]")
    if not (x.flags.c_contiguous and y.flags.c_contiguous):
        raise ValueError("axpy needs C-contiguous x and y (the kernel walks raw buffers)")
    prec = _precision(y)
    kernel(prec, "var_axpy")((blocks(y.size),), (256,), (y, x, y.dtype.type(a), np.int64(y.size)))


def axpy_neg_device(s, x, y) -> None:
    """``y = y + a x`` in place with ``a = -s[0]``, ``s`` a :func:`dot_device` result."""
    if x.dtype != y.dtype or x.size != y.size:
        raise ValueError(f"axpy of {x.dtype}[{x.size}] into {y.dtype}[{y.size}]")
    if not (x.flags.c_contiguous and y.flags.c_contiguous):
        raise ValueError("axpy needs C-contiguous x and y (the kernel walks raw buffers)")
    prec = _precision(y)
    kernel(prec, "var_axpy_neg_dev")((blocks(y.size),), (256,), (y, x, s, np.int64(y.size)))
