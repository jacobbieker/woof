"""In-place writes into facade storage.

CUDA buffers keep their address for their whole life, and woof builds
device-pointer tables from ``ndarray.data.ptr``; so a write into an existing
array must change its bytes, not swap in a new buffer.  Every write
(``a[...] = v``, ``fill``, ``copyto``, ``out=``, ``+=``) runs one strided
copy kernel through the same in-place launcher as translated kernels,
choosing the kernel by element width so any dtype, view and stride works.
"""

from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np

MAX_DIMS = 8

_SOURCE = """#include <metal_stdlib>
struct WoofCopy { long n; long offset; long ndim; long shape[8]; long stride[8]; };
#define WOOF_COPY(NAME, T)                                                              \\
[[kernel]] void NAME(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]],    \\
                     constant WoofCopy& c [[buffer(2)]],                                \\
                     uint3 tid [[thread_position_in_threadgroup]],                      \\
                     uint3 bid [[threadgroup_position_in_grid]],                        \\
                     uint3 bd [[threads_per_threadgroup]]) {                            \\
  long i = (long)bid.x * (long)bd.x + (long)tid.x;                                     \\
  if (i >= c.n) return;                                                                 \\
  long off = c.offset;                                                                  \\
  if (c.n <= 0xFFFFFFFFl) {   /* 32-bit division where the index fits */                \\
    uint rem = uint(i);                                                                 \\
    for (long d = c.ndim - 1; d > 0; --d) {                                             \\
      uint s = uint(c.shape[d]), k = rem % s; rem /= s; off += long(k) * c.stride[d];   \\
    }                                                                                   \\
    off += long(rem) * c.stride[0];                                                     \\
  } else {                                                                              \\
    long rem = i;                                                                       \\
    for (long d = c.ndim - 1; d > 0; --d) {                                             \\
      long k = rem % c.shape[d]; rem /= c.shape[d]; off += k * c.stride[d];             \\
    }                                                                                   \\
    off += rem * c.stride[0];                                                           \\
  }                                                                                     \\
  dst[off] = src[SRC_INDEX];                                                            \\
}
#define SRC_INDEX i
WOOF_COPY(woof_copy_1, uchar)
WOOF_COPY(woof_copy_2, ushort)
WOOF_COPY(woof_copy_4, uint)
WOOF_COPY(woof_copy_8, ulong)
#undef SRC_INDEX
#define SRC_INDEX 0
WOOF_COPY(woof_fill_1, uchar)
WOOF_COPY(woof_fill_2, ushort)
WOOF_COPY(woof_fill_4, uint)
WOOF_COPY(woof_fill_8, ulong)
"""
_KEY = "woof:facade-strided-copy:v2"
_TPB = 256


@lru_cache(maxsize=1)
def _compiled():
    from woof.backend.metal import ext
    ext.load().compile(_KEY, _SOURCE, [f"woof_{kind}_{w}" for kind in ("copy", "fill")
                                       for w in (1, 2, 4, 8)], 0)
    return True


def _width_dtype(itemsize: int):
    import mlx.core as mx
    return {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[itemsize]


def _collapsed(shape, strides):
    """The same row-major walk over fewer dimensions: length-1 axes dropped
    and each axis merged into the next where it steps over exactly that
    axis.  The kernel divides once per inner axis (64-bit division, which
    Metal emulates), so a contiguous view takes none."""
    dims = [(n, s) for n, s in zip(shape, strides) if n != 1] or [(1, 0)]
    out = [dims[-1]]
    for n, s in reversed(dims[:-1]):
        inner_n, inner_s = out[-1]
        if s == inner_s * inner_n:
            out[-1] = (n * inner_n, inner_s)
        else:
            out.append((n, s))
    out.reverse()
    return [n for n, _ in out], [s for _, s in out]


def write(view, values) -> None:
    """Store ``values`` (an mx.array in the view's storage representation,
    broadcastable to its shape) into ``view`` in place."""
    import mlx.core as mx
    from woof.backend.facade._array import flush_pending, order_launch
    from woof.backend.metal import ext
    item = view.itemsize
    if item not in (1, 2, 4, 8):
        raise NotImplementedError(f"in-place writes of {item}-byte elements")
    if view.ndim > MAX_DIMS:
        raise NotImplementedError(f"in-place writes into {view.ndim}-dimensional views")
    n = view.size
    if n == 0:
        return
    shape = view.shape or (1,)
    # One value for every element (fill, ``a[...] = 0``): the fill kernel
    # reads it in every thread instead of a materialized broadcast copy.
    fill = values.size == 1
    src = mx.contiguous(values.reshape(-1) if fill else
                        mx.broadcast_to(values, view.shape).reshape(-1))
    src = src.view(_width_dtype(item)) if src.dtype.size == item else \
        src.astype(values.dtype).view(_width_dtype(item))
    if view._offset % item or any(s % item for s in view.strides):
        raise NotImplementedError("a view whose offset or strides are not whole elements")
    shape, strides = _collapsed(shape, [s // item for s in (view.strides or (item,))])
    blob = struct.pack("<qqq", n, view._offset // item, len(shape))
    blob += struct.pack(f"<{MAX_DIMS}q", *(list(shape) + [1] * (MAX_DIMS - len(shape))))
    blob += struct.pack(f"<{MAX_DIMS}q", *(strides + [0] * (MAX_DIMS - len(strides))))
    _compiled()
    st = view._st
    order_launch((), (st,))
    flush_pending()
    kernel = f"woof_{'fill' if fill else 'copy'}_{item}"
    (out,) = ext.load().launch(_KEY, _SOURCE, kernel, ((n + _TPB - 1) // _TPB, 1, 1),
                               (_TPB, 1, 1), [st.arr, src], [0, 0], [True, False], [0, 1], -1,
                               [blob], [2], 0, 0)
    st.rebind_written(out)
