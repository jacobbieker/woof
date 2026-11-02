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


class ndarray(np.ndarray):
    """CuPy's ndarray as a NumPy array in host memory."""

    def __new__(cls, shape, dtype=float, memptr=None, strides=None, order="C"):
        dtype = np.dtype(dtype)
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
        if memptr is not None:
            # CuPy's ndarray over an existing allocation (MemoryPointer)
            count = int(np.prod(shape)) if shape else 1
            if strides is not None:
                span = (sum((n - 1) * abs(s) for n, s in zip(shape, strides)) + dtype.itemsize
                        if count else 0)
            else:
                span = count * dtype.itemsize
            raw = (ctypes.c_char * builtins.max(span, 1)).from_address(int(memptr.ptr))
            arr = np.ndarray.__new__(cls, shape, dtype, buffer=raw, strides=strides, order=order)
            arr._owner = memptr
            return arr
        return np.ndarray.__new__(cls, shape, dtype, strides=strides, order=order)

    def __array_finalize__(self, obj):
        return None

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        # CuPy returns 0-d arrays where NumPy returns scalars (reductions)
        arr = arr.view(type(self)) if isinstance(arr, np.ndarray) else np.asarray(arr).view(type(self))
        return arr

    def __getitem__(self, key):
        r = np.ndarray.__getitem__(self, key)
        if isinstance(r, np.generic):
            return np.asarray(r).view(ndarray)
        return r

    def get(self, stream=None, order="C", out=None, blocking=True):
        host = np.array(self, order=order, subok=False, copy=True)
        if out is not None:
            out[...] = host
            return out
        return host

    def set(self, arr, stream=None):
        self[...] = np.asarray(arr)

    @property
    def data(self):
        from woof.backend.hostfacade.cuda import MemoryPointer, _memory_of
        mem, offset = _memory_of(self)
        return MemoryPointer(mem, offset)

    @property
    def device(self):
        from woof.backend.hostfacade.cuda import Device
        return Device(0)

    def reduced_view(self, dtype=None):
        return self

    def toDlpack(self):
        return np.asarray(self).__dlpack__()

    def __reduce__(self):
        # pickles as the host array it is (as CuPy's pickles its values)
        return (_unpickle, (np.array(self, subok=False),))


def _unpickle(host):
    return host.view(ndarray)


def _up(r):
    """A NumPy result as the facade's: arrays as :class:`ndarray`."""
    if type(r) is np.ndarray:
        return r.view(ndarray)
    if isinstance(r, tuple):
        return tuple(_up(x) for x in r)
    if isinstance(r, list):
        return [_up(x) for x in r]
    return r


def _wrap(fn):
    def call(*args, **kwargs):
        return _up(fn(*args, **kwargs))
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
    return np.array(obj, dtype=dtype, copy=True, order=order, ndmin=ndmin).view(ndarray)


def asarray(a, dtype=None, order=None, blocking=False):
    if isinstance(a, ndarray):
        if (dtype is None or np.dtype(dtype) == a.dtype) and (
                order in (None, "K", "A") or (order == "C" and a.flags.c_contiguous)
                or (order == "F" and a.flags.f_contiguous)):
            return a
        return np.asarray(a, dtype=dtype, order=order).view(ndarray) if order else \
            a.astype(dtype, copy=False)
    # host data: a copy, as an upload is
    return np.array(a, dtype=dtype, order=order or "K", copy=True).view(ndarray)


asanyarray = asarray


def ascontiguousarray(a, dtype=None):
    return np.ascontiguousarray(asarray(a), dtype=dtype).view(ndarray)


def asfortranarray(a, dtype=None):
    return np.asfortranarray(asarray(a), dtype=dtype).view(ndarray)


def empty(shape, dtype=float, order="C"):
    return np.empty(shape, dtype=dtype, order=order).view(ndarray)


def zeros(shape, dtype=float, order="C"):
    return np.zeros(shape, dtype=dtype, order=order).view(ndarray)


def ones(shape, dtype=float, order="C"):
    return np.ones(shape, dtype=dtype, order=order).view(ndarray)


def full(shape, fill_value, dtype=None, order="C"):
    return np.full(shape, fill_value, dtype=dtype, order=order).view(ndarray)


def _like(np_fn):
    def fn(a, *args, dtype=None, order="K", subok=None, shape=None, **kw):
        return np_fn(np.asarray(a), *args, dtype=dtype, order=order, shape=shape, **kw).view(ndarray)
    fn.__name__ = np_fn.__name__
    return fn


empty_like, zeros_like, ones_like, full_like = (_like(np.empty_like), _like(np.zeros_like),
                                                _like(np.ones_like), _like(np.full_like))


def get_array_module(*args):
    import sys
    return sys.modules[__name__] if builtins.any(isinstance(a, ndarray) for a in args) else np


def fromDlpack(capsule):
    return np.from_dlpack(capsule).view(ndarray)


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
        return _up(target(*[np.asarray(x) if isinstance(x, ndarray) else x for x in a], **k))
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
    if isinstance(value, (np.ufunc, type)) or not callable(value):
        return value
    wrapped = _wrap(value)
    globals()[name] = wrapped
    return wrapped
