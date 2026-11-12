"""Zero (or copy) the outer frame of a field's horizontal planes in one launch.

Open-boundary strips (``dycore._zero_open_strips``) and the specified ring of
a PD advective tendency (``moist._exclude_specified_ring_advection``) were
each four slice assignments, ``buf[..., :w] = 0`` and so on: four launches
per call, about 90 a step at 3 km.  :func:`zero_frame` stores +0 at exactly
the same elements in one launch; :func:`copy_frame` copies them from a
second array of the same shape (the held boundary tendency's ring).
"""
from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np


@lru_cache(maxsize=None)
def _frame_kernel(copy: bool):
    body = "\n".join([
        "const long long plane = (long long)ny * nx;",
        "const int col = (int)(i % nx);",
        "const int row = (int)((i % plane) / nx);",
        "if (col < x_lo || col >= nx - x_hi || row < y_lo || row >= ny - y_hi) "
        + ("buf[i] = src[i];" if copy else "buf[i] = T(0);"),
    ])
    return cp.ElementwiseKernel(
        "int32 nx, int32 ny, int32 x_lo, int32 x_hi, int32 y_lo, int32 y_hi"
        + (", raw T src" if copy else ""),
        "raw T buf", body, "gpuwm_copy_frame" if copy else "gpuwm_zero_frame")


def _launch(buf, src, x_lo, x_hi, y_lo, y_hi) -> bool:
    if buf.ndim < 2 or buf.size == 0 or not buf.flags.c_contiguous:
        return False
    if src is not None and (src.shape != buf.shape or src.dtype != buf.dtype
                            or not src.flags.c_contiguous):
        return False
    ny, nx = buf.shape[-2:]

    def clamp(v, n):
        return max(0, min(int(v), n))

    args = [np.int32(nx), np.int32(ny), np.int32(clamp(x_lo, nx)),
            np.int32(clamp(x_hi, nx)), np.int32(clamp(y_lo, ny)),
            np.int32(clamp(y_hi, ny))]
    if src is not None:
        args.append(src)
    _frame_kernel(src is not None)(*args, buf, size=buf.size)
    return True


def zero_frame(buf, x_lo: int, x_hi: int, y_lo: int, y_hi: int) -> bool:
    """Store +0 at every element of ``buf`` (``(..., ny, nx)``) whose column
    is below ``x_lo`` or among the last ``x_hi``, or whose row is below
    ``y_lo`` or among the last ``y_hi``.  This is the same set as
    ``buf[..., :x_lo] = 0; buf[..., nx - x_hi:] = 0; buf[..., :y_lo, :] = 0;
    buf[..., ny - y_hi:, :] = 0``.

    Returns False and writes nothing for a buffer the kernel does not
    cover: not C-contiguous, fewer than two dimensions, or empty.  The
    caller then assigns the slices itself.
    """
    return _launch(buf, None, x_lo, x_hi, y_lo, y_hi)


def copy_frame(buf, src, x_lo: int, x_hi: int, y_lo: int, y_hi: int) -> bool:
    """:func:`zero_frame`'s elements of ``buf`` take ``src``'s values
    (``src``: the same shape and dtype, C-contiguous, not overlapping
    ``buf``).  False: nothing written, as for :func:`zero_frame`."""
    return _launch(buf, src, x_lo, x_hi, y_lo, y_hi)
