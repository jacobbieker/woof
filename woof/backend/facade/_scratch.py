"""``cupy.empty`` without the fill, for scratch a kernel writes before it reads.

The facade's ``empty`` is ``zeros``: MLX has no uninitialized allocation, so
every ``cp.empty`` costs a GPU pass over the whole buffer (3.4 ms per GiB on
an M-series GPU), where CUDA's ``empty`` costs nothing and holds garbage.  A
buffer whose every read word is written first (a kernel's private arena) gets
the same words either way.  ``uninitialized`` allocates the buffer as the
output of a kernel that writes nothing (``mx.fast.metal_kernel`` with no init
value): one empty dispatch instead of a fill.
"""

from __future__ import annotations

import functools
import operator

import numpy as np

# Never true (the input is one 0): the output is bound and left as allocated.
_SOURCE = "if (thread_position_in_grid.x == 0 && flag[0] != 0) out[0] = 0;"


@functools.lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    flag = mx.zeros((1,), dtype=mx.int32)
    mx.eval(flag)
    kernel = mx.fast.metal_kernel(name="woof_uninitialized", input_names=["flag"],
                                  output_names=["out"], source=_SOURCE)
    return kernel, flag


def uninitialized(shape, dtype=np.float32):
    """A C-contiguous facade array of ``shape`` whose words are unspecified."""
    from woof.backend.facade._array import (MAX_ELEMENTS, Storage, ndarray, storage_dtype,
                                            storage_mx_dtype)
    from woof.backend.facade._ops import empty
    dtype = np.dtype(dtype)
    shape = (int(shape),) if isinstance(shape, (int, np.integer)) else tuple(int(n) for n in shape)
    n = functools.reduce(operator.mul, shape, 1)
    words = n * dtype.itemsize // storage_dtype(dtype).itemsize
    if words == 0 or n > MAX_ELEMENTS or words > MAX_ELEMENTS:
        return empty(shape, dtype)          # the ordinary path (and its refusal)
    kernel, flag = _kernel()
    (arr,) = kernel(inputs=[flag], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                    output_shapes=[(words,)], output_dtypes=[storage_mx_dtype(dtype)])
    # A kernel output is a flat, row-contiguous buffer it owns: held as it is
    # (no contiguous copy), and queued like any new storage (PENDING).
    return ndarray(shape, dtype, _storage=Storage(arr, dtype, ready=True))
