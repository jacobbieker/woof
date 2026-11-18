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

# Never true (n is the element count): the output is bound and left as allocated.
_SOURCE = "if (thread_position_in_grid.x == 0 && n[0] < 0) out[0] = 0;"


@functools.lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name="woof_uninitialized", input_names=["n"],
                                output_names=["out"], source=_SOURCE)


def uninitialized(shape, dtype=np.float32):
    """A C-contiguous facade array of ``shape`` whose words are unspecified."""
    import mlx.core as mx
    from woof.backend.facade._array import (MAX_ELEMENTS, Storage, ndarray, storage_dtype,
                                            storage_mx_dtype)
    from woof.backend.facade._ops import empty
    dtype = np.dtype(dtype)
    shape = (int(shape),) if isinstance(shape, (int, np.integer)) else tuple(int(n) for n in shape)
    n = functools.reduce(operator.mul, shape, 1)
    words = n * dtype.itemsize // storage_dtype(dtype).itemsize
    if words == 0 or n > MAX_ELEMENTS:
        return empty(shape, dtype)          # the ordinary path (and its refusal)
    (arr,) = _kernel()(inputs=[mx.array([words], dtype=mx.int64)], grid=(1, 1, 1),
                       threadgroup=(1, 1, 1), output_shapes=[(words,)],
                       output_dtypes=[storage_mx_dtype(dtype)])
    return ndarray(shape, dtype, _storage=Storage(arr, dtype, ready=True))
