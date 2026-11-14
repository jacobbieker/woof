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
    flush_pending()
    st = view._st
    kernel = f"woof_{'fill' if fill else 'copy'}_{item}"
    (out,) = ext.load().launch(_KEY, _SOURCE, kernel, ((n + _TPB - 1) // _TPB, 1, 1),
                               (_TPB, 1, 1), [st.arr, src], [0, 0], [True, False], [0, 1], -1,
                               [blob], [2], 0, 0)
    st.rebind_written(out)


#: ``a op= b`` in fast numerics (binary32): one kernel reading both operands
#: and writing ``a`` in place, where it was MLX's op into a new array plus a
#: strided copy.  Metal's safe-math ``+ - * /`` give MLX's words, subnormal
#: flushing included (checked word for word over random and subnormal-biased
#: operands), so the result is the same; the kernel is built with safe math.
_ARITH_SOURCE = """#include <metal_stdlib>
struct WoofArith { long n; long dst_offset; long src_offset; long ndim;
                   long shape[8]; long dst_stride[8]; long src_stride[8]; };
#define WOOF_ARITH(NAME, OP)                                                            \\
[[kernel]] void NAME(device float* dst [[buffer(0)]], const device float* src [[buffer(1)]], \\
                     constant WoofArith& c [[buffer(2)]],                               \\
                     uint3 tid [[thread_position_in_threadgroup]],                      \\
                     uint3 bid [[threadgroup_position_in_grid]],                        \\
                     uint3 bd [[threads_per_threadgroup]]) {                            \\
  long i = (long)bid.x * (long)bd.x + (long)tid.x;                                     \\
  if (i >= c.n) return;                                                                 \\
  long d = c.dst_offset, s = c.src_offset;                                              \\
  if (c.n <= 0xFFFFFFFFl) {   /* 32-bit division where the index fits */                \\
    uint rem = uint(i);                                                                 \\
    for (long a = c.ndim - 1; a > 0; --a) {                                             \\
      uint m = uint(c.shape[a]), k = rem % m; rem /= m;                                 \\
      d += long(k) * c.dst_stride[a]; s += long(k) * c.src_stride[a];                   \\
    }                                                                                   \\
    d += long(rem) * c.dst_stride[0]; s += long(rem) * c.src_stride[0];                 \\
  } else {                                                                              \\
    long rem = i;                                                                       \\
    for (long a = c.ndim - 1; a > 0; --a) {                                             \\
      long k = rem % c.shape[a]; rem /= c.shape[a];                                     \\
      d += k * c.dst_stride[a]; s += k * c.src_stride[a];                               \\
    }                                                                                   \\
    d += rem * c.dst_stride[0]; s += rem * c.src_stride[0];                             \\
  }                                                                                     \\
  dst[d] = dst[d] OP src[s];                                                            \\
}
WOOF_ARITH(woof_iadd_f32, +)
WOOF_ARITH(woof_isub_f32, -)
WOOF_ARITH(woof_imul_f32, *)
WOOF_ARITH(woof_idiv_f32, /)
"""
_ARITH_KEY = "woof:facade-inplace-arith:v1"
_ARITH_KERNELS = {"add": "woof_iadd_f32", "subtract": "woof_isub_f32",
                  "multiply": "woof_imul_f32", "divide": "woof_idiv_f32",
                  "true_divide": "woof_idiv_f32"}


@lru_cache(maxsize=1)
def _arith_compiled():
    from woof.backend.metal import ext
    ext.load().compile(_ARITH_KEY, _ARITH_SOURCE, sorted(set(_ARITH_KERNELS.values())), 0)
    return True


def _collapsed_pair(shape, dst, src):
    """:func:`_collapsed` over two stride sets walked together."""
    dims = [(n, a, b) for n, a, b in zip(shape, dst, src) if n != 1] or [(1, 0, 0)]
    out = [dims[-1]]
    for n, a, b in reversed(dims[:-1]):
        inner_n, inner_a, inner_b = out[-1]
        if a == inner_a * inner_n and b == inner_b * inner_n:
            out[-1] = (n * inner_n, inner_a, inner_b)
        else:
            out.append((n, a, b))
    out.reverse()
    return [d[0] for d in out], [d[1] for d in out], [d[2] for d in out]


def arith_in_place(np_name, view, other) -> bool:
    """``view op= other`` as one kernel, for a binary32 ``view`` (any strides)
    and a binary32 facade array or a Python/NumPy scalar whose result type is
    binary32.  False (nothing done) where the general path must run: another
    op or dtype, an ``other`` sharing the view's buffer, or a shape the
    operands do not broadcast to."""
    import mlx.core as mx
    from woof.backend.facade._array import flush_pending, ndarray
    from woof.backend.metal import ext
    kernel = _ARITH_KERNELS.get(np_name)
    if kernel is None or view.dtype != np.float32 or view.ndim > MAX_DIMS:
        return False
    n = view.size
    if n == 0:
        return False
    if isinstance(other, ndarray):
        if other.dtype != np.float32 or other.ndim > view.ndim:
            return False
        if other._st.buffer_root() is view._st.buffer_root():
            return False
        try:
            if np.broadcast_shapes(view.shape, other.shape) != view.shape:
                return False
        except ValueError:
            return False
        lead = view.ndim - other.ndim
        o_shape = (1,) * lead + other.shape
        o_strides = (0,) * lead + other.strides
        if other._offset % 4 or any(s % 4 for s in other.strides):
            return False
        src_strides = [0 if m == 1 and v != 1 else s // 4
                       for m, v, s in zip(o_shape, view.shape, o_strides)]
        src_offset = other._offset // 4
        src = other._st.arr
    elif isinstance(other, (int, float, np.floating, np.integer)) and not isinstance(other, (bool, np.bool_)):
        try:
            if np.result_type(view.dtype, other if isinstance(other, (int, float)) else other.dtype) \
                    != np.float32:
                return False
        except TypeError:
            return False
        src = mx.array(np.asarray([other], dtype=np.float32))
        src_strides, src_offset = [0] * view.ndim, 0
    else:
        return False
    if view._offset % 4 or any(s % 4 for s in view.strides):
        return False
    if any(s == 0 and m > 1 for s, m in zip(view.strides, view.shape)):
        return False   # a broadcast destination aliases itself; the general path writes it once
    shape = view.shape or (1,)
    dst_strides = [s // 4 for s in view.strides] or [1]
    src_strides = src_strides or [0]
    shape, dst_strides, src_strides = _collapsed_pair(shape, dst_strides, src_strides)
    pad = [0] * (MAX_DIMS - len(shape))
    blob = struct.pack("<qqqq", n, view._offset // 4, src_offset, len(shape))
    blob += struct.pack(f"<{MAX_DIMS}q", *(list(shape) + [1] * len(pad)))
    blob += struct.pack(f"<{MAX_DIMS}q", *(dst_strides + pad))
    blob += struct.pack(f"<{MAX_DIMS}q", *(src_strides + pad))
    _arith_compiled()
    flush_pending()
    st = view._st
    (out,) = ext.load().launch(_ARITH_KEY, _ARITH_SOURCE, kernel, ((n + _TPB - 1) // _TPB, 1, 1),
                               (_TPB, 1, 1), [st.arr, src], [0, 0], [True, False], [0, 1], -1,
                               [blob], [2], 0, 0)
    st.rebind_written(out)
    return True
