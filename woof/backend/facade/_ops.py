"""Array operations of the facade: creation, conversion, element-wise
arithmetic, reductions.

Result dtypes follow NumPy 2's rules (``np.result_type`` with Python scalars
weak), which is what CuPy 14 follows.  Arithmetic involving binary64 runs on
the host through NumPy, because MLX's GPU has no float64; everything else
runs on MLX.

Word-for-word parity of these ops with CuPy's own ElementwiseKernels is
graded per op by the facade receipt (tools/facade_op_receipt.py); an op whose
MLX kernel is not word-exact (sqrt, the transcendental functions) is a known
delta until it is routed to a woof Metal kernel.
"""

from __future__ import annotations

import numpy as np

from woof.backend.facade._array import ndarray, storage_mx_dtype, to_storage_values


def _mx():
    import mlx.core as mx
    return mx


# ---------------------------------------------------------------- values

def _dtype_hint(x):
    if isinstance(x, ndarray):
        return x.dtype
    if isinstance(x, np.ndarray):
        return x.dtype
    if isinstance(x, np.generic):
        return x.dtype
    if isinstance(x, (bool, int, float, complex)):
        return x
    return np.asarray(x).dtype


def scalar_values(value, dtype):
    """A Python/NumPy scalar (or a one-element facade array) as a 0-d
    mx.array in ``dtype``'s storage."""
    if isinstance(value, ndarray):
        if value.size != 1:
            raise ValueError(f"a fill value must be a scalar; got shape {value.shape}")
        value = value.get().reshape(())
    host = np.asarray(value, dtype=dtype)
    return to_storage_values(host).reshape(())


def operand_values(value, dtype):
    """Any operand as an mx.array in ``dtype``'s storage representation."""
    mx = _mx()
    dtype = np.dtype(dtype)
    if isinstance(value, ndarray):
        if value.dtype == dtype:
            return value._values()
        return astype(value, dtype)._values()
    host = np.asarray(value)
    if host.dtype != dtype:
        host = host.astype(dtype)
    return to_storage_values(host).reshape(host.shape)


def _wrap(values, dtype, *, fresh: bool = False) -> ndarray:
    """A new array holding ``values`` (storage representation), in its own
    storage; ``fresh`` when ``values`` was just computed (no copy needed)."""
    from woof.backend.facade._array import Storage
    shape = tuple(values.shape)
    return ndarray(shape, dtype, _storage=Storage(values.astype(storage_mx_dtype(dtype)), dtype,
                                                  fresh=fresh))


def from_host(host: np.ndarray) -> ndarray:
    # (np.ascontiguousarray promotes a 0-d array to 1-d: keep the scalar shape)
    host = np.require(np.asarray(host), requirements="C")
    if host.dtype == np.float64 or host.dtype.kind in "biufc?":
        return _wrap(to_storage_values(host).reshape(host.shape), host.dtype, fresh=True)
    raise TypeError(f"the mlx backend cannot hold dtype {host.dtype}")


def _host(x):
    return x.get() if isinstance(x, ndarray) else x


# ---------------------------------------------------------------- creation

def empty(shape, dtype=np.float32, order="C"):
    if order == "F":
        return _fortran_from_host(np.zeros(shape, dtype=dtype, order="F"))
    return ndarray(shape, dtype)


def zeros(shape, dtype=np.float32, order="C"):
    return empty(shape, dtype, order)


def ones(shape, dtype=np.float32, order="C"):
    return full(shape, 1, dtype, order)


def full(shape, fill_value, dtype=None, order="C"):
    import mlx.core as mx
    if dtype is None:
        dtype = fill_value.dtype if isinstance(fill_value, ndarray) else np.asarray(fill_value).dtype
    shape = (int(shape),) if isinstance(shape, (int, np.integer)) else tuple(int(n) for n in shape)
    if order == "F" and len(shape) > 1:
        return full(tuple(reversed(shape)), fill_value, dtype).transpose()
    return _wrap(mx.broadcast_to(scalar_values(fill_value, dtype), shape), np.dtype(dtype), fresh=True)


def _like(a, dtype, shape, order="K"):
    """dtype, shape and memory order of a ``*_like`` result: "K" and "A"
    keep a column-major prototype's layout, which a kernel indexing the
    buffer relies on as it does on CUDA."""
    if order in ("K", "A", None):
        f = (isinstance(a, ndarray) and a.ndim > 1 and a.is_f_contiguous()
             and not a.is_c_contiguous() and (shape is None or len(tuple(np.atleast_1d(shape))) == a.ndim))
        order = "F" if f else "C"
    return (a.dtype if dtype is None else dtype), (a.shape if shape is None else shape), order


def empty_like(a, dtype=None, order="K", subok=None, shape=None):
    dtype, shape, order = _like(a, dtype, shape, order)
    return empty(shape, dtype, order)


def zeros_like(a, dtype=None, order="K", subok=None, shape=None):
    dtype, shape, order = _like(a, dtype, shape, order)
    return zeros(shape, dtype, order)


def ones_like(a, dtype=None, order="K", subok=None, shape=None):
    dtype, shape, order = _like(a, dtype, shape, order)
    return ones(shape, dtype, order)


def full_like(a, fill_value, dtype=None, order="K", subok=None, shape=None):
    dtype, shape, order = _like(a, dtype, shape, order)
    return full(shape, fill_value, dtype, order)


def arange(*args, dtype=None):
    return from_host(np.arange(*args, dtype=dtype))


def linspace(*args, **kwargs):
    return from_host(np.linspace(*args, **kwargs))


def asarray(a, dtype=None, order=None, *, blocking=False):
    if isinstance(a, ndarray):
        a = a if dtype is None or np.dtype(dtype) == a.dtype else astype(a, dtype)
        return asfortranarray(a) if order == "F" else a
    host = np.asarray(a, dtype=dtype)
    if order == "F" or (order in (None, "K", "A") and host.ndim > 1 and host.flags.f_contiguous
                        and not host.flags.c_contiguous):
        return _fortran_from_host(host)
    return from_host(host)


def array(a, dtype=None, copy=True, order="K", subok=False, ndmin=0, *, blocking=False):
    """NumPy's layout rule: ``order`` C or F is that layout; K and A keep a
    Fortran-contiguous source's column-major memory (kernels index
    ``cp.array(x, order="F")`` buffers as Fortran arrays)."""
    if isinstance(a, ndarray):
        fortran = order == "F" or (order in ("K", "A", None) and a.ndim > 1
                                   and a.is_f_contiguous() and not a.is_c_contiguous())
        out = a if dtype is None else astype(a, dtype)
        if fortran:
            return _fortran_from_host(out.get())
        if not out.is_c_contiguous():
            return out.copy()
        return out.copy() if copy and out is a else out
    host = np.array(a, dtype=dtype, ndmin=ndmin, order=order if order in ("C", "F", "A", "K") else "K")
    if host.ndim > 1 and host.flags.f_contiguous and not host.flags.c_contiguous:
        return _fortran_from_host(host)
    return from_host(host)


def ascontiguousarray(a, dtype=None):
    a = asarray(a, dtype)
    return a if a.is_c_contiguous() else a.copy()


def asfortranarray(a, dtype=None):
    """Column-major storage, with strides to match (NumPy's model)."""
    a = asarray(a, dtype)
    if a.ndim <= 1 or a.is_f_contiguous():
        return a if a.ndim > 1 else ascontiguousarray(a)
    return _fortran_from_host(a.get())


def _fortran_from_host(host: np.ndarray) -> ndarray:
    from woof.backend.facade._array import Storage
    host = np.asfortranarray(host)
    flat = host.ravel(order="F")
    st = Storage(to_storage_values(flat), host.dtype, fresh=True)
    return ndarray(host.shape, host.dtype, _storage=st, _strides=host.strides)


def asnumpy(a, stream=None, order="C", out=None, blocking=True):
    if isinstance(a, ndarray):
        return a.get(out=out)
    return np.asarray(a)


def astype(a: ndarray, dtype) -> ndarray:
    """CuPy's ``astype`` (order "K"): a column-major array stays column-major."""
    if a.ndim > 1 and a._memory_order("K") == "F":
        return astype(a.transpose(), dtype).transpose()
    mx = _mx()
    dtype = np.dtype(dtype)
    if a.dtype == np.float64 or dtype == np.float64:
        return from_host(a.get().astype(dtype))
    return _wrap(a._values().astype(storage_mx_dtype(dtype)), dtype)


def copyto(dst, src, casting="same_kind", where=None):
    if where is not None:
        src = where_(where, src, dst)
    dst._write(operand_values(src, dst.dtype))


# ---------------------------------------------------------------- element-wise

def _apply(np_name, mx_name, *args, out=None, dtype=None, result_dtype=None):
    hints = [_dtype_hint(x) for x in args]
    if result_dtype is None:
        try:
            result_dtype = np.result_type(*hints)
        except TypeError:
            result_dtype = np.result_type(*[np.asarray(h).dtype if not isinstance(h, np.dtype) else h
                                            for h in hints])
    if dtype is not None:
        result_dtype = np.dtype(dtype)
    involves_f64 = (np.dtype(result_dtype) == np.float64
                    or any(isinstance(h, np.dtype) and h == np.float64 for h in hints))
    mx = _mx()
    if involves_f64 or mx_name is None or not hasattr(mx, mx_name):
        host = getattr(np, np_name)(*[_host(x) for x in args])
        if dtype is not None:
            host = np.asarray(host).astype(dtype)
        result = from_host(np.asarray(host))
    else:
        operands = []
        for x in args:
            if isinstance(x, ndarray):
                operands.append(x._values())
            elif isinstance(x, np.ndarray):
                operands.append(to_storage_values(x).reshape(x.shape))
            elif isinstance(x, np.generic):
                operands.append(mx.array(x))
            else:
                operands.append(x)
        values = getattr(mx, mx_name)(*operands)
        if not isinstance(values, mx.array):
            values = mx.array(values)
        rdt = np.dtype(result_dtype)
        values = values.astype(storage_mx_dtype(rdt))
        # an operation's result is a new buffer, unless MLX handed an input back
        inputs = [x._st.arr for x in args if isinstance(x, ndarray)]
        result = _wrap(values, rdt, fresh=not any(values is arr for arr in inputs))
    if out is not None:
        copyto(out, result)
        return out
    return result


_BINARY = {
    "add": "add", "subtract": "subtract", "multiply": "multiply", "divide": "divide",
    "true_divide": "divide", "floor_divide": "floor_divide", "power": "power",
    "maximum": "maximum", "minimum": "minimum", "fmax": "maximum", "fmin": "minimum",
    "remainder": "remainder", "mod": "remainder", "arctan2": "arctan2",
    "bitwise_and": "bitwise_and", "bitwise_or": "bitwise_or", "bitwise_xor": "bitwise_xor",
    "left_shift": "left_shift", "right_shift": "right_shift", "hypot": None,
    "copysign": None, "logaddexp": "logaddexp", "nextafter": None, "ldexp": None,
    "float_power": None, "gcd": None, "lcm": None, "heaviside": None,
}
_COMPARE = {
    "less": "less", "less_equal": "less_equal", "greater": "greater",
    "greater_equal": "greater_equal", "equal": "equal", "not_equal": "not_equal",
    "logical_and": "logical_and", "logical_or": "logical_or",
}
_UNARY = {
    "negative": "negative", "abs": "abs", "absolute": "abs", "fabs": "abs", "sqrt": "sqrt",
    "exp": "exp", "log": "log", "log10": "log10", "log2": "log2", "log1p": "log1p",
    "expm1": "expm1", "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "arcsin",
    "arccos": "arccos", "arctan": "arctan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "floor": "floor", "ceil": "ceil", "trunc": None, "rint": "round", "round": "round",
    "sign": "sign", "square": "square", "reciprocal": "reciprocal", "deg2rad": "radians",
    "rad2deg": "degrees", "cbrt": None, "exp2": None, "erf": "erf",
}
_PREDICATE = {"isfinite": "isfinite", "isnan": "isnan", "isinf": "isinf",
              "logical_not": "logical_not", "signbit": None}


def _binary(np_name):
    mx_name = _BINARY[np_name]

    def fn(a, b, out=None, dtype=None, **_):
        return _apply(np_name, mx_name, a, b, out=out, dtype=dtype)
    fn.__name__ = np_name
    return fn


def _compare(np_name):
    mx_name = _COMPARE[np_name]

    def fn(a, b, out=None, **_):
        return _apply(np_name, mx_name, a, b, out=out, result_dtype=np.bool_)
    fn.__name__ = np_name
    return fn


def _unary(np_name):
    mx_name = _UNARY[np_name]

    def fn(a, out=None, dtype=None, **_):
        return _apply(np_name, mx_name, a, out=out, dtype=dtype)
    fn.__name__ = np_name
    return fn


def _predicate(np_name):
    mx_name = _PREDICATE[np_name]

    def fn(a, out=None, **_):
        return _apply(np_name, mx_name, a, out=out, result_dtype=np.bool_)
    fn.__name__ = np_name
    return fn


class _BinaryUfunc:
    """A binary ufunc with NumPy's methods.  The call runs on the device;
    ``reduce``, ``reduceat``, ``accumulate`` and ``outer`` run NumPy's on the
    host words, so they are exact and slow."""

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = fn.__name__
        self.nin, self.nout = 2, 1

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def _host_method(self, method, *args, out=None, **kwargs):
        r = from_host(np.asarray(getattr(getattr(np, self.__name__), method)(
            *[_host(x) for x in args], **kwargs)))
        if out is not None:
            copyto(out, r)
            return out
        return r

    def reduce(self, a, axis=0, dtype=None, out=None, keepdims=False):
        return self._host_method("reduce", a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def reduceat(self, a, indices, axis=0, dtype=None, out=None):
        return self._host_method("reduceat", a, indices, axis=axis, dtype=dtype, out=out)

    def accumulate(self, a, axis=0, dtype=None, out=None):
        return self._host_method("accumulate", a, axis=axis, dtype=dtype, out=out)

    def outer(self, a, b, **kwargs):
        return self._host_method("outer", a, b, **kwargs)


UFUNCS = {}
UFUNCS.update({n: _BinaryUfunc(_binary(n)) for n in _BINARY})
UFUNCS.update({n: _compare(n) for n in _COMPARE})
UFUNCS.update({n: _unary(n) for n in _UNARY})
UFUNCS.update({n: _predicate(n) for n in _PREDICATE})


def where_(cond, x=None, y=None):
    if x is None and y is None:
        return tuple(from_host(i) for i in np.nonzero(_host(cond)))
    result_dtype = np.result_type(_dtype_hint(x), _dtype_hint(y))
    mx = _mx()
    if result_dtype == np.float64:
        return from_host(np.where(_host(cond), _host(x), _host(y)))
    c = operand_values(cond, np.bool_)
    xv = operand_values(x, result_dtype)
    yv = operand_values(y, result_dtype)
    return _wrap(mx.where(c, xv, yv), result_dtype)


def clip(a, a_min, a_max, out=None):
    r = a
    if a_min is not None:
        r = UFUNCS["maximum"](r, a_min)
    if a_max is not None:
        r = UFUNCS["minimum"](r, a_max)
    if out is not None:
        copyto(out, r)
        return out
    return r


# ---------------------------------------------------------------- reductions

def _reduce(np_name, mx_name, a, axis=None, dtype=None, out=None, keepdims=False, **_):
    a = asarray(a)
    mx = _mx()
    if a.dtype == np.float64 or (dtype is not None and np.dtype(dtype) == np.float64) or mx_name is None:
        host = getattr(np, np_name)(a.get(), axis=axis, keepdims=keepdims,
                                    **({"dtype": dtype} if dtype is not None else {}))
        r = from_host(np.asarray(host))
    else:
        values = a._values()
        if dtype is not None:
            values = values.astype(storage_mx_dtype(np.dtype(dtype)))
        kw = {"keepdims": keepdims}
        if axis is not None:
            kw["axis"] = axis
        v = getattr(mx, mx_name)(values, **kw)
        if dtype is not None and np_name in ("sum", "prod", "mean"):
            rdt = np.dtype(dtype)       # NumPy's: an explicit dtype is the result's
        else:
            rdt = np.result_type(getattr(np, np_name)(np.zeros((1,), dtype or a.dtype)))
        r = _wrap(v.astype(storage_mx_dtype(rdt)), rdt)
    if out is not None:
        copyto(out, r)
        return out
    return r


def _reduction(np_name, mx_name):
    def fn(a, axis=None, dtype=None, out=None, keepdims=False, **kw):
        return _reduce(np_name, mx_name, a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    fn.__name__ = np_name
    return fn


REDUCTIONS = {
    "sum": _reduction("sum", "sum"), "prod": _reduction("prod", "prod"),
    "max": _reduction("max", "max"), "min": _reduction("min", "min"),
    "amax": _reduction("max", "max"), "amin": _reduction("min", "min"),
    "mean": _reduction("mean", "mean"), "any": _reduction("any", "any"),
    "all": _reduction("all", "all"), "nanmax": _reduction("nanmax", None),
    "nanmin": _reduction("nanmin", None), "std": _reduction("std", None),
    "var": _reduction("var", None),
}


def count_nonzero(a, axis=None):
    a = asarray(a)
    return _reduce("sum", "sum", UFUNCS["not_equal"](a, 0), axis=axis, dtype=np.int64)


def argmax(a, axis=None):
    return from_host(np.asarray(np.argmax(_host(a), axis=axis)))


def argmin(a, axis=None):
    return from_host(np.asarray(np.argmin(_host(a), axis=axis)))


def concatenate(arrays, axis=0, out=None, dtype=None):
    arrays = [asarray(x) for x in arrays]
    rdt = np.result_type(*[x.dtype for x in arrays]) if dtype is None else np.dtype(dtype)
    if rdt == np.float64:
        r = from_host(np.concatenate([x.get() for x in arrays], axis=axis).astype(rdt))
    else:
        mx = _mx()
        r = _wrap(mx.concatenate([operand_values(x, rdt) for x in arrays], axis=axis), rdt)
    if out is not None:
        copyto(out, r)
        return out
    return r


def stack(arrays, axis=0, out=None):
    arrays = [asarray(x) for x in arrays]
    rdt = np.result_type(*[x.dtype for x in arrays])
    if rdt == np.float64:
        r = from_host(np.stack([x.get() for x in arrays], axis=axis))
    else:
        mx = _mx()
        r = _wrap(mx.stack([operand_values(x, rdt) for x in arrays], axis=axis), rdt)
    if out is not None:
        copyto(out, r)
        return out
    return r


def broadcast_to(a, shape):
    a = asarray(a)
    shape = tuple(int(n) for n in shape)
    nd = len(shape)
    src_shape = (1,) * (nd - a.ndim) + a.shape
    src_strides = (0,) * (nd - a.ndim) + a.strides
    strides = tuple(0 if s == 1 and n != 1 else st for s, n, st in zip(src_shape, shape, src_strides))
    for s, n in zip(src_shape, shape):
        if s != 1 and s != n:
            raise ValueError(f"cannot broadcast {a.shape} to {shape}")
    return ndarray(shape, a.dtype, _storage=a._st, _offset=a._offset, _strides=strides)


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return bool(np.allclose(_host(a), _host(b), rtol=rtol, atol=atol, equal_nan=equal_nan))


# ---------------------------------------------------------------- ndarray methods

def _inplace(name):
    fn = UFUNCS[name]

    def method(self, other):
        self._write(operand_values(fn(self, other), self.dtype))
        return self
    return method


_DUNDER = {
    "__add__": ("add", False), "__radd__": ("add", True),
    "__sub__": ("subtract", False), "__rsub__": ("subtract", True),
    "__mul__": ("multiply", False), "__rmul__": ("multiply", True),
    "__truediv__": ("divide", False), "__rtruediv__": ("divide", True),
    "__floordiv__": ("floor_divide", False), "__rfloordiv__": ("floor_divide", True),
    "__pow__": ("power", False), "__rpow__": ("power", True),
    "__mod__": ("remainder", False), "__rmod__": ("remainder", True),
    "__and__": ("bitwise_and", False), "__rand__": ("bitwise_and", True),
    "__or__": ("bitwise_or", False), "__ror__": ("bitwise_or", True),
    "__xor__": ("bitwise_xor", False), "__rxor__": ("bitwise_xor", True),
    "__lshift__": ("left_shift", False), "__rshift__": ("right_shift", False),
    "__lt__": ("less", False), "__le__": ("less_equal", False),
    "__gt__": ("greater", False), "__ge__": ("greater_equal", False),
    "__eq__": ("equal", False), "__ne__": ("not_equal", False),
}

for _name, (_uf, _reflected) in _DUNDER.items():
    def _make(uf=_uf, reflected=_reflected):
        fn = UFUNCS[uf]
        if reflected:
            return lambda self, other: fn(other, self)
        return lambda self, other: fn(self, other)
    setattr(ndarray, _name, _make())

for _name, _uf in {"__iadd__": "add", "__isub__": "subtract", "__imul__": "multiply",
                   "__itruediv__": "divide", "__ifloordiv__": "floor_divide",
                   "__ipow__": "power", "__imod__": "remainder", "__iand__": "bitwise_and",
                   "__ior__": "bitwise_or", "__ixor__": "bitwise_xor"}.items():
    setattr(ndarray, _name, _inplace(_uf))

ndarray.__neg__ = lambda self: UFUNCS["negative"](self)
ndarray.__pos__ = lambda self: self.copy()
ndarray.__abs__ = lambda self: UFUNCS["abs"](self)
ndarray.__invert__ = lambda self: _apply("invert", "bitwise_invert", self)
ndarray.__hash__ = None
# a @ b: NumPy's matmul on the host words (cupy.matmul in the facade)
ndarray.__matmul__ = lambda self, other: from_host(np.matmul(self.get(), _host(other)))
ndarray.__rmatmul__ = lambda self, other: from_host(np.matmul(_host(other), self.get()))

def _scan_out(r, out):
    if out is None:
        return r
    copyto(out, r)
    return out


# .real / .imag: the array itself and zeros for a real dtype; the parts,
# computed on the host words, for a complex one
ndarray.real = property(lambda self: from_host(np.ascontiguousarray(self.get().real))
                        if self.dtype.kind == "c" else self)
ndarray.imag = property(lambda self: from_host(np.ascontiguousarray(self.get().imag))
                        if self.dtype.kind == "c" else zeros(self.shape, self.dtype))

METHODS = {name: fn for name, fn in REDUCTIONS.items()}
METHODS.update({
    "clip": clip,
    "argmax": argmax, "argmin": argmin,
    "nonzero": lambda a: where_(a),
    "round": lambda a, decimals=0: UFUNCS["round"](a) if decimals == 0 else
    from_host(np.round(a.get(), decimals)),
    "cumsum": lambda a, axis=None, dtype=None, out=None: _scan_out(
        from_host(np.cumsum(a.get(), axis=axis, dtype=dtype)), out),
    "dot": lambda a, b: from_host(np.dot(a.get(), _host(b))),
    "take": lambda a, indices, axis=None, out=None, mode="raise": _scan_out(
        from_host(np.take(a.get(), _host(indices), axis=axis, mode=mode)), out),
})
