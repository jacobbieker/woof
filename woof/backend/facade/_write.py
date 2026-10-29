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
  long rem = i, off = c.offset;                                                         \\
  for (long d = c.ndim - 1; d >= 0; --d) {                                              \\
    long k = rem % c.shape[d]; rem /= c.shape[d]; off += k * c.stride[d];               \\
  }                                                                                     \\
  dst[off] = src[i];                                                                    \\
}
WOOF_COPY(woof_copy_1, uchar)
WOOF_COPY(woof_copy_2, ushort)
WOOF_COPY(woof_copy_4, uint)
WOOF_COPY(woof_copy_8, ulong)
"""
_KEY = "woof:facade-strided-copy:v1"
_TPB = 256


@lru_cache(maxsize=1)
def _compiled():
    from woof.backend.metal import ext
    ext.load().compile(_KEY, _SOURCE, [f"woof_copy_{w}" for w in (1, 2, 4, 8)], 0)
    return True


def _width_dtype(itemsize: int):
    import mlx.core as mx
    return {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[itemsize]


def write(view, values) -> None:
    """Store ``values`` (an mx.array in the view's storage representation,
    broadcastable to its shape) into ``view`` in place."""
    import mlx.core as mx
    from woof.backend.facade._array import flush_pending
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
    src = mx.contiguous(mx.broadcast_to(values, view.shape).reshape(-1))
    src = src.view(_width_dtype(item)) if src.dtype.size == item else \
        src.astype(values.dtype).view(_width_dtype(item))
    if view._offset % item or any(s % item for s in view.strides):
        raise NotImplementedError("a view whose offset or strides are not whole elements")
    strides = view.strides or (item,)
    blob = struct.pack("<qqq", n, view._offset // item, len(shape))
    blob += struct.pack(f"<{MAX_DIMS}q", *(list(shape) + [1] * (MAX_DIMS - len(shape))))
    blob += struct.pack(f"<{MAX_DIMS}q", *([s // item for s in strides] + [0] * (MAX_DIMS - len(strides))))
    _compiled()
    flush_pending()
    st = view._st
    (out,) = ext.load().launch(_KEY, _SOURCE, f"woof_copy_{item}", ((n + _TPB - 1) // _TPB, 1, 1),
                               (_TPB, 1, 1), [st.arr, src], [0, 0], [True, False], [0, 1], -1,
                               [blob], [2], 0, 0)
    st.rebind_written(out)
