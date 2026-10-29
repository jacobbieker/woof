"""NumPy indexing on facade arrays.

Basic indexing (integers, slices, ``...``, ``None``) is a view: new shape,
strides and offset over the same storage, as NumPy and CuPy do.  Anything
else (integer arrays, boolean masks) gathers on read and scatters on write.
"""

from __future__ import annotations

import numpy as np

from woof.backend.facade._array import ndarray


def _is_basic(key) -> bool:
    items = key if isinstance(key, tuple) else (key,)
    return all(isinstance(k, (int, np.integer, slice)) or k is Ellipsis or k is None
               for k in items)


def basic_view(a: ndarray, key) -> ndarray:
    items = list(key) if isinstance(key, tuple) else [key]
    n_index = sum(1 for k in items if k is not None and k is not Ellipsis)
    if items.count(Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if Ellipsis in items:
        at = items.index(Ellipsis)
        items[at:at + 1] = [slice(None)] * (a.ndim - n_index)
    else:
        items += [slice(None)] * (a.ndim - n_index)
    if sum(1 for k in items if k is not None) != a.ndim:
        raise IndexError(f"too many indices for array: array is {a.ndim}-dimensional")
    shape, strides, offset, axis = [], [], a._offset, 0
    for k in items:
        if k is None:
            shape.append(1)
            strides.append(0)
            continue
        n, s = a.shape[axis], a.strides[axis]
        if isinstance(k, (int, np.integer)):
            k = int(k)
            if k < -n or k >= n:
                raise IndexError(f"index {k} is out of bounds for axis {axis} with size {n}")
            offset += (k % n) * s
        else:
            start, stop, step = k.indices(n)
            length = len(range(start, stop, step))
            shape.append(length)
            strides.append(s * step)
            if length:
                offset += start * s
        axis += 1
    return ndarray(tuple(shape), a.dtype, _storage=a._st, _offset=offset, _strides=tuple(strides))


def _advanced_key(key):
    """Translate facade arrays inside a key to mx arrays for gather/scatter."""
    import mlx.core as mx
    items = key if isinstance(key, tuple) else (key,)
    out = []
    for k in items:
        if isinstance(k, ndarray):
            v = k._values()
            if k.dtype == np.bool_:
                # mx has no boolean-mask indexing; use the flat positions.
                host = k.get()
                v = mx.array(np.flatnonzero(host).astype(np.int64)) if host.ndim == 1 else None
                if v is None:
                    raise NotImplementedError("multi-dimensional boolean mask indexing")
            out.append(v)
        elif isinstance(k, np.ndarray):
            if k.dtype == np.bool_:
                if k.ndim != 1:
                    raise NotImplementedError("multi-dimensional boolean mask indexing")
                k = np.flatnonzero(k)
            out.append(mx.array(k.astype(np.int64)))
        elif isinstance(k, list):
            out.append(mx.array(np.asarray(k, dtype=np.int64)))
        else:
            out.append(k)
    return tuple(out) if isinstance(key, tuple) else out[0]


def getitem(a: ndarray, key):
    if _is_basic(key):
        return basic_view(a, key)
    values = a._values()[_advanced_key(key)]
    out = ndarray(tuple(values.shape), a.dtype)
    out._write(values)
    return out


def setitem(a: ndarray, key, value):
    from woof.backend.facade._ops import operand_values
    if _is_basic(key):
        view = basic_view(a, key)
        view._write(operand_values(value, a.dtype))
        return
    # Advanced: scatter into the full view, then write the view back.
    whole = a._values()
    whole[_advanced_key(key)] = operand_values(value, a.dtype)
    a._write(whole)
