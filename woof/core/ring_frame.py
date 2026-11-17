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
    """One thread per frame element, not per buffer element: element ``i``
    of the launch is the ``r``-th frame element of plane ``i / per_plane``,
    counted over the ``y_lo`` top rows, then the ``y_hi`` bottom rows, then
    each middle row's ``side`` columns (its ``x_lo`` first and ``x_hi``
    last, or all ``nx`` when they meet).  Unsigned 32-bit index arithmetic:
    the caller launches only when every index fits."""
    body = "\n".join([
        "const unsigned int j = (unsigned int)i;",
        "const unsigned int plane = j / (unsigned int)per_plane;",
        "unsigned int r = j - plane * (unsigned int)per_plane;",
        "const unsigned int w = (unsigned int)nx;",
        "const unsigned int top = (unsigned int)y_lo * w;",
        "const unsigned int bottom = (unsigned int)y_hi * w;",
        "unsigned int row, col;",
        "if (r < top) { row = r / w; col = r - row * w; }",
        "else if (r - top < bottom) { r -= top; const unsigned int k = r / w;"
        " row = (unsigned int)(ny - y_hi) + k; col = r - k * w; }",
        "else { r -= top + bottom; const unsigned int k = r / (unsigned int)side;"
        " const unsigned int c = r - k * (unsigned int)side; row = (unsigned int)y_lo + k;"
        " col = (side == nx || c < (unsigned int)x_lo) ? c : (unsigned int)(nx - x_hi) + (c - (unsigned int)x_lo); }",
        "const unsigned int e = plane * (unsigned int)ny * w + row * w + col;",
        "buf[e] = src[e];" if copy else "buf[e] = T(0);",
    ])
    return cp.ElementwiseKernel(
        "int32 nx, int32 ny, int32 x_lo, int32 x_hi, int32 y_lo, int32 y_hi, int32 side, int32 per_plane"
        + (", raw T src" if copy else ""),
        "raw T buf", body, "gpuwm_copy_frame" if copy else "gpuwm_zero_frame")


def _launch(buf, src, x_lo, x_hi, y_lo, y_hi) -> bool:
    if buf.ndim < 2 or buf.size == 0 or not buf.flags.c_contiguous:
        return False
    if src is not None and (src.shape != buf.shape or src.dtype != buf.dtype
                            or not src.flags.c_contiguous):
        return False
    ny, nx = buf.shape[-2:]
    if buf.size >= 2 ** 31:
        return False                  # the kernel's indices are 32-bit

    def clamp(v, n):
        return max(0, min(int(v), n))

    x_lo, x_hi = clamp(x_lo, nx), clamp(x_hi, nx)
    y_lo, y_hi = clamp(y_lo, ny), clamp(y_hi, ny)
    if y_lo + y_hi >= ny:             # every row is a frame row
        y_lo, y_hi = ny, 0
    side = nx if x_lo + x_hi >= nx else x_lo + x_hi
    per_plane = (y_lo + y_hi) * nx + (ny - y_lo - y_hi) * side
    count = (buf.size // (ny * nx)) * per_plane
    if count == 0:
        return True                   # an empty frame: nothing to store
    args = [np.int32(nx), np.int32(ny), np.int32(x_lo), np.int32(x_hi), np.int32(y_lo),
            np.int32(y_hi), np.int32(side), np.int32(per_plane)]
    if src is not None:
        args.append(src)
    _frame_kernel(src is not None)(*args, buf, size=count)
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
