"""A CuPy-compatible module for the ``host`` backend, over NumPy.

:func:`woof.backend.install` makes ``import cupy`` resolve here under
``WOOF_BACKEND=host``.  "Device" arrays are :class:`ndarray`, a NumPy
subclass in host memory: ``isinstance(x, cupy.ndarray)`` tells them from
host arrays as it does under CuPy, ``x.get()`` copies out, ``x.data.ptr`` is
the address kernels receive.  Array operations are NumPy's.  Raw kernels
(``RawModule``/``RawKernel``) and ``ElementwiseKernel``/``ReductionKernel``
are the same CUDA source compiled by the host C++ compiler
(:mod:`woof.backend.host.loader`).

The MLX facade (:mod:`woof.backend.facade`) is the same surface over Apple's
GPU; this module is independent of it so that neither needs MLX or a GPU
for the other.
"""

from __future__ import annotations

import builtins
import ctypes
import types

import numpy as np

__woof_facade__ = True
__version__ = "14.0.0+woof.host"


class BackendUnsupported(AttributeError):
    """A CuPy attribute the host facade does not implement."""


class ndarray:
    """CuPy's ndarray over a NumPy array in host memory (``_a``).

    Not a NumPy subclass: woof tells device arrays from host ones with
    ``isinstance(x, np.ndarray)`` (and ``__cuda_array_interface__``), as it
    does under CuPy.  NumPy's protocols (``__array_ufunc__``,
    ``__array_function__``) run NumPy on the host array and hand facade
    arrays back, so ``np.where(m, x, y)`` on facade arrays is one, as it is
    on CuPy's."""

    __slots__ = ("_a", "_owner", "__weakref__")
    __array_priority__ = 100

    def __init__(self, shape, dtype=float, memptr=None, strides=None, order="C"):
        dtype = np.dtype(dtype)
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
        self._owner = None
        if memptr is None:
            self._a = np.ndarray(shape, dtype, strides=strides, order=order)
            return
        # CuPy's ndarray over an existing allocation (a MemoryPointer)
        count = int(np.prod(shape)) if shape else 1
        if strides is not None:
            span = sum((n - 1) * abs(st) for n, st in zip(shape, strides)) + dtype.itemsize if count else 0
        else:
            span = count * dtype.itemsize
        raw = (ctypes.c_char * builtins.max(span, 1)).from_address(int(memptr.ptr))
        self._a = np.ndarray(shape, dtype, buffer=raw, strides=strides, order=order)
        self._owner = memptr

    @classmethod
    def _of(cls, a):
        self = cls.__new__(cls)
        self._a = a
        self._owner = None
        return self

    # -- NumPy's view of it: the host array itself
    @property
    def __array_interface__(self):
        return self._a.__array_interface__

    @property
    def __cuda_array_interface__(self):
        """The array interface, as CuPy's arrays carry it: woof also asks
        ``hasattr(x, "__cuda_array_interface__")`` to tell device arrays."""
        iface = dict(self._a.__array_interface__)
        iface["version"] = 3
        iface.setdefault("strides", None)
        return iface

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self._a, dtype=dtype, copy=True)
        return self._a if dtype is None else self._a.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if out is not None:
            kwargs["out"] = tuple(_down(o) for o in out)
        result = getattr(ufunc, method)(*_down(inputs), **_down(kwargs))
        if out is not None:
            return out[0] if len(out) == 1 else out
        return _up(result, scalars=True)

    def __array_function__(self, func, types, args, kwargs):
        return _up(func(*_down(args), **_down(kwargs)), scalars=True)

    # -- the attributes CuPy adds
    def get(self, stream=None, order="C", out=None, blocking=True):
        host = np.array(self._a, order=order, copy=True)
        if out is not None:
            out[...] = host
            return out
        return host

    def set(self, arr, stream=None):
        self._a[...] = _down(arr)

    @property
    def data(self):
        from woof.backend.hostfacade.cuda import MemoryPointer, _memory_of
        mem, offset = _memory_of(self)
        return MemoryPointer(mem, offset, keep=self)

    @property
    def device(self):
        from woof.backend.hostfacade.cuda import Device
        return Device(0)

    def reduced_view(self, dtype=None):
        return self

    def toDlpack(self):
        return self._a.__dlpack__()

    def __dlpack__(self, *args, **kwargs):
        return self._a.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._a.__dlpack_device__()

    # -- NumPy's own attributes and methods, results as facade arrays
    shape = property(lambda self: self._a.shape)
    dtype = property(lambda self: self._a.dtype)
    ndim = property(lambda self: self._a.ndim)
    size = property(lambda self: self._a.size)
    nbytes = property(lambda self: self._a.nbytes)
    itemsize = property(lambda self: self._a.itemsize)
    strides = property(lambda self: self._a.strides)
    flags = property(lambda self: self._a.flags)
    T = property(lambda self: ndarray._of(self._a.T))
    base = property(lambda self: _up(self._a.base))

    @shape.setter
    def shape(self, value):
        self._a.shape = value

    @property
    def real(self):
        return ndarray._of(self._a.real)

    @property
    def imag(self):
        return ndarray._of(self._a.imag)

    def __getattr__(self, name):
        value = getattr(self._a, name)
        if callable(value):
            def method(*args, **kwargs):
                return _up(value(*_down(args), **_down(kwargs)), scalars=name not in _SCALAR_METHODS)
            method.__name__ = name
            return method
        return _up(value)

    def __getitem__(self, key):
        return _up(self._a[_down(key)], scalars=True)

    def __setitem__(self, key, value):
        self._a[_down(key)] = _down(value)

    def __len__(self):
        return len(self._a)

    def __iter__(self):
        for row in self._a:
            yield _up(row, scalars=True)

    def __bool__(self):
        return bool(self._a)

    def __float__(self):
        return float(self._a)

    def __int__(self):
        return int(self._a)

    def __index__(self):
        return self._a.__index__()

    def __complex__(self):
        return complex(self._a)

    def __round__(self, ndigits=None):
        return _up(round(self._a, ndigits) if ndigits is not None else self._a.round(), scalars=True)

    def __format__(self, spec):
        return format(self._a, spec)

    def __contains__(self, value):
        return _down(value) in self._a

    def __repr__(self):
        return repr(self._a)

    def __str__(self):
        return str(self._a)

    def __copy__(self):
        return ndarray._of(self._a.copy())

    def __deepcopy__(self, memo):
        return ndarray._of(self._a.copy())

    def __reduce__(self):
        # pickles as its values (CuPy's arrays pickle their values too)
        return (_unpickle, (np.array(self._a, copy=True),))

    __hash__ = None


#: Methods whose NumPy results are host values in CuPy too.
_SCALAR_METHODS = frozenset(("item", "tolist", "tobytes", "tofile", "dumps", "dump", "__index__"))


def _binary(ufunc):
    def op(self, other):
        return ufunc(self, other)

    def rop(self, other):
        return ufunc(other, self)

    def iop(self, other):
        ufunc(self._a, _down(other), out=self._a)
        return self
    return op, rop, iop


for _name, _ufunc in (("add", np.add), ("sub", np.subtract), ("mul", np.multiply),
                      ("truediv", np.true_divide), ("floordiv", np.floor_divide), ("mod", np.remainder),
                      ("pow", np.power), ("lshift", np.left_shift), ("rshift", np.right_shift),
                      ("and", np.bitwise_and), ("or", np.bitwise_or), ("xor", np.bitwise_xor),
                      ("matmul", np.matmul)):
    _op, _rop, _iop = _binary(_ufunc)
    setattr(ndarray, f"__{_name}__", _op)
    setattr(ndarray, f"__r{_name}__", _rop)
    setattr(ndarray, f"__i{_name}__", _iop)
for _name, _ufunc in (("eq", np.equal), ("ne", np.not_equal), ("lt", np.less), ("le", np.less_equal),
                      ("gt", np.greater), ("ge", np.greater_equal)):
    setattr(ndarray, f"__{_name}__", _binary(_ufunc)[0])
for _name, _ufunc in (("neg", np.negative), ("pos", np.positive), ("abs", np.absolute),
                      ("invert", np.invert)):
    setattr(ndarray, f"__{_name}__", lambda self, _u=_ufunc: _u(self))
ndarray.__divmod__ = lambda self, other: np.divmod(self, other)


def _unpickle(host):
    return ndarray._of(host)


def _down(x):
    """Facade arrays inside ``x`` (tuples, lists, dicts) as their host arrays."""
    if isinstance(x, ndarray):
        return x._a
    if isinstance(x, tuple):
        return tuple(_down(y) for y in x)
    if isinstance(x, list):
        return [_down(y) for y in x]
    if isinstance(x, dict):
        return {k: _down(v) for k, v in x.items()}
    return x


def _up(r, scalars=False):
    """A NumPy result as the facade's: arrays as :class:`ndarray`, and with
    ``scalars`` a NumPy scalar as a 0-d array (CuPy returns arrays where
    NumPy returns scalars: reductions, element reads)."""
    if isinstance(r, np.ndarray):
        return ndarray._of(r)
    if scalars and isinstance(r, np.generic):
        return ndarray._of(np.asarray(r))
    if isinstance(r, tuple):
        return tuple(_up(x, scalars) for x in r)
    if isinstance(r, list):
        return [_up(x, scalars) for x in r]
    return r


class _UFunc:
    """A NumPy ufunc whose results are facade arrays (CuPy's ufuncs return
    device arrays whatever the operands)."""

    def __init__(self, ufunc):
        self._u = ufunc
        self.__name__ = ufunc.__name__

    def __call__(self, *args, **kwargs):
        out = kwargs.get("out")
        r = self._u(*_down(args), **_down(kwargs))
        if out is not None:
            return out
        return _up(r, scalars=True)

    def __getattr__(self, name):
        value = getattr(self._u, name)
        return _wrap(value) if callable(value) else value


def _wrap(fn):
    def call(*args, **kwargs):
        return _up(fn(*_down(args), **_down(kwargs)), scalars=True)
    call.__name__ = getattr(fn, "__name__", "fn")
    call.__doc__ = f"NumPy's {call.__name__} (woof host facade)."
    call.__wrapped__ = fn
    return call


# ---------------------------------------------------------------- creation and transfer

def asnumpy(a, stream=None, order="C", out=None, blocking=True):
    if isinstance(a, ndarray):
        return a.get(order=order, out=out)
    host = np.asarray(a, order=None if order == "K" else order)
    if out is not None:
        out[...] = host
        return out
    return host


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0, blocking=False):
    if isinstance(obj, ndarray) and not copy and (dtype is None or np.dtype(dtype) == obj.dtype):
        return obj
    return ndarray._of(np.array(_down(obj), dtype=dtype, copy=True, order=order, ndmin=ndmin))


def asarray(a, dtype=None, order=None, blocking=False):
    if isinstance(a, ndarray):
        h = a._a
        if (dtype is None or np.dtype(dtype) == h.dtype) and (
                order in (None, "K", "A") or (order == "C" and h.flags.c_contiguous)
                or (order == "F" and h.flags.f_contiguous)):
            return a
        return ndarray._of(np.asarray(h, dtype=dtype, order=order))
    # host data: a copy, as an upload is
    return ndarray._of(np.array(_down(a), dtype=dtype, order=order or "K", copy=True))


asanyarray = asarray


def ascontiguousarray(a, dtype=None):
    a = asarray(a)
    h = np.ascontiguousarray(a._a, dtype=dtype)
    return a if h is a._a else ndarray._of(h)


def asfortranarray(a, dtype=None):
    a = asarray(a)
    h = np.asfortranarray(a._a, dtype=dtype)
    return a if h is a._a else ndarray._of(h)


def empty(shape, dtype=float, order="C"):
    return ndarray._of(np.empty(shape, dtype=dtype, order=order))


def zeros(shape, dtype=float, order="C"):
    return ndarray._of(np.zeros(shape, dtype=dtype, order=order))


def ones(shape, dtype=float, order="C"):
    return ndarray._of(np.ones(shape, dtype=dtype, order=order))


def full(shape, fill_value, dtype=None, order="C"):
    return ndarray._of(np.full(shape, _down(fill_value), dtype=dtype, order=order))


def _like(np_fn):
    def fn(a, *args, dtype=None, order="K", subok=None, shape=None, **kw):
        return ndarray._of(np_fn(_down(a), *_down(args), dtype=dtype, order=order, shape=shape, **kw))
    fn.__name__ = np_fn.__name__
    return fn


empty_like, zeros_like, ones_like, full_like = (_like(np.empty_like), _like(np.zeros_like),
                                                _like(np.ones_like), _like(np.full_like))


def get_array_module(*args):
    import sys
    return sys.modules[__name__] if builtins.any(isinstance(a, ndarray) for a in args) else np


def fromDlpack(capsule):
    return ndarray._of(np.from_dlpack(capsule))


from_dlpack = fromDlpack


def clear_memo():
    return None


def fuse(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda f: f


# dtypes: NumPy's own, as CuPy's are.
float16, float32, float64 = np.float16, np.float32, np.float64
int8, int16, int32, int64 = np.int8, np.int16, np.int32, np.int64
uint8, uint16, uint32, uint64 = np.uint8, np.uint16, np.uint32, np.uint64
bool_, complex64, complex128 = np.bool_, np.complex64, np.complex128
intp, uintp = np.intp, np.uintp
dtype = np.dtype
result_type, promote_types, can_cast = np.result_type, np.promote_types, np.can_cast
finfo, iinfo = np.finfo, np.iinfo
nan, inf, pi, e, newaxis = np.nan, np.inf, np.pi, np.e, None
isscalar = np.isscalar
generic, number, integer, floating, signedinteger, unsignedinteger = (
    np.generic, np.number, np.integer, np.floating, np.signedinteger, np.unsignedinteger)

# ---------------------------------------------------------------- submodules over NumPy / SciPy


class _Namespace(types.ModuleType):
    """A CuPy submodule as NumPy's, results as facade arrays."""

    def __init__(self, name, target):
        super().__init__(name)
        self._target = target

    def __getattr__(self, name):
        value = getattr(self._target, name)
        if isinstance(value, np.ufunc) or isinstance(value, type):
            return value
        if callable(value):
            return _wrap(value)
        return value


linalg = _Namespace("cupy.linalg", np.linalg)
fft = _Namespace("cupy.fft", np.fft)
fft.config = types.SimpleNamespace(enable_nd_planning=False, get_plan_cache=lambda: None)
random = _Namespace("cupy.random", np.random)


def _testing_on_host(fn):
    def check(*args, **kwargs):
        return fn(*[asnumpy(a) if isinstance(a, ndarray) else a for a in args], **kwargs)
    check.__name__ = fn.__name__
    return check


testing = types.SimpleNamespace(**{n: _testing_on_host(getattr(np.testing, n)) for n in (
    "assert_array_equal", "assert_array_almost_equal", "assert_allclose", "assert_array_less",
    "assert_array_max_ulp", "assert_array_almost_equal_nulp", "assert_equal")})

from woof.backend.hostfacade import cuda  # noqa: E402

get_default_memory_pool = cuda.get_default_memory_pool
get_default_pinned_memory_pool = cuda.get_default_pinned_memory_pool


# ---------------------------------------------------------------- raw kernels

class RawModule:
    """``cupy.RawModule(code=..., options=...)`` built by the host compiler."""

    def __init__(self, code=None, path=None, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, name_expressions=None, jitify=False):
        if code is None and path is not None:
            with open(path, encoding="utf-8") as f:
                code = f.read()
        if code is None:
            raise ValueError("RawModule needs code= or path=")
        import hashlib
        from woof.backend.host.loader import load_unit
        name = "rawmodule-" + hashlib.sha256(code.encode()).hexdigest()[:12]
        self._module = load_unit(name, code, options=tuple(options or ()),
                                 name_expressions=name_expressions)
        self.code = code
        self.options = tuple(options or ())

    def get_function(self, name):
        return self._module.get_function(name)

    def get_global(self, name):
        return self._module.get_global(name)

    get_global_var = get_global

    def compile(self, log_stream=None):
        return None


class RawKernel:
    """``cupy.RawKernel(code, name, options=...)``."""

    def __init__(self, code, name, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, jitify=False):
        self._function = RawModule(code=code, options=options).get_function(name)
        self.code = code
        self.name = name
        self.options = tuple(options or ())

    @property
    def attributes(self):
        return self._function.attributes

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        return self._function(grid, block, args, shared_mem=shared_mem, stream=stream)

    def compile(self, log_stream=None):
        return None


from woof.backend.hostfacade._elementwise import ElementwiseKernel, ReductionKernel  # noqa: E402

# ---------------------------------------------------------------- cupyx

cupyx = types.ModuleType("cupyx")
cupyx.__woof_facade__ = True
cupyx.__path__ = []


def _cupyx_getattr(name):
    raise BackendUnsupported(f"cupyx.{name} is not implemented by woof's host backend")


cupyx.__getattr__ = _cupyx_getattr
cupyx.empty_pinned = lambda shape, dtype=np.float64, order="C": np.empty(shape, dtype=dtype, order=order)
cupyx.zeros_pinned = lambda shape, dtype=np.float64, order="C": np.zeros(shape, dtype=dtype, order=order)
cupyx.scatter_add = lambda a, slices, value: np.add.at(a, slices, value)
cupyx.get_runtime_info = lambda *a, **k: "woof host backend"


def _scipy_on_host(module, name):
    def fn(*a, **k):
        import importlib
        target = getattr(importlib.import_module(f"scipy.{module}"), name)
        return _up(target(*_down(a), **_down(k)))
    fn.__name__ = name
    return fn


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__path__ = []
    return m


cupyx.scipy = _module("cupyx.scipy",
    special=_module("cupyx.scipy.special", **{n: _scipy_on_host("special", n)
                                              for n in ("erf", "erfc", "gamma", "gammaln")}),
    ndimage=_module("cupyx.scipy.ndimage", **{n: _scipy_on_host("ndimage", n) for n in (
        "minimum_filter", "maximum_filter", "uniform_filter", "gaussian_filter", "label", "convolve",
        "map_coordinates", "zoom", "median_filter", "binary_dilation", "binary_erosion",
        "distance_transform_edt")}))
#: the facade's own submodules, for `import cupyx.scipy.special` (backend._FacadeFinder)
SUBMODULES = {"cupyx.scipy": cupyx.scipy, "cupyx.scipy.special": cupyx.scipy.special,
              "cupyx.scipy.ndimage": cupyx.scipy.ndimage, "cupy.linalg": linalg, "cupy.fft": fft,
              "cupy.random": random, "cupy.testing": testing}


def configure(kind: str):
    if kind != "host":
        raise BackendUnsupported(f"the host facade serves backend 'host', not {kind!r}")


# ---------------------------------------------------------------- everything else: NumPy's

_NOT_CUPY = frozenset(("matrix", "char", "rec", "ctypeslib", "lib", "ma", "polynomial", "f2py",
                       "distutils", "core", "emath", "typing", "exceptions", "dtypes"))


def __getattr__(name):
    if name.startswith("__") or name in _NOT_CUPY:
        raise AttributeError(name)
    try:
        value = getattr(np, name)
    except AttributeError:
        raise BackendUnsupported(f"cupy.{name} is not implemented by woof's host backend") from None
    if isinstance(value, type) or not callable(value):
        return value
    wrapped = _UFunc(value) if isinstance(value, np.ufunc) else _wrap(value)
    globals()[name] = wrapped
    return wrapped
