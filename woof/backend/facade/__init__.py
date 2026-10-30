"""A CuPy-compatible module for the ``mlx`` (and ``cpu``) backends.

:func:`woof.backend.install` puts this module in ``sys.modules["cupy"]``, so
every ``import cupy as cp`` in woof gets it and runs unchanged.  It
implements the part of CuPy's API that woof uses (the census in
tools/cupy_surface_census.py lists it); any other attribute raises
:class:`BackendUnsupported` naming it, so a gap fails loudly at its first use
instead of degrading.

Arrays are :class:`woof.backend.facade._array.ndarray`; raw kernels
(``RawModule``/``RawKernel``) are the same CUDA source translated to Metal
(:mod:`woof.backend.translate`, :mod:`woof.backend.metal.loader`).
"""

from __future__ import annotations

import builtins
import types

import numpy as np

from woof.backend.facade import cuda
from woof.backend.facade._array import ndarray
from woof.backend.facade._ops import (  # noqa: F401  -- the public surface
    REDUCTIONS, UFUNCS, allclose, arange, argmax, argmin, array, ascontiguousarray,
    asfortranarray, asnumpy, asarray, broadcast_to, clip, concatenate, copyto,
    count_nonzero, empty, empty_like, full, full_like, linspace, ones, ones_like,
    stack, where_ as where, zeros, zeros_like,
)

__woof_facade__ = True
__version__ = "14.0.0+woof.facade"

_kind: str | None = None


class BackendUnsupported(AttributeError):
    """A CuPy attribute the woof facade does not implement."""


def configure(kind: str):
    global _kind
    if kind != "mlx":
        raise BackendUnsupported(f"the facade backend {kind!r} is not implemented yet")
    from woof.backend.metal.loader import set_wired_limit, share_one_stream
    set_wired_limit()
    share_one_stream()
    _kind = kind


# dtypes: NumPy's own, as CuPy's are.
float16, float32, float64 = np.float16, np.float32, np.float64
int8, int16, int32, int64 = np.int8, np.int16, np.int32, np.int64
uint8, uint16, uint32, uint64 = np.uint8, np.uint16, np.uint32, np.uint64
bool_, complex64, complex128 = np.bool_, np.complex64, np.complex128
dtype = np.dtype
result_type, promote_types, can_cast = np.result_type, np.promote_types, np.can_cast
finfo, iinfo = np.finfo, np.iinfo
nan, inf, pi, e, newaxis = np.nan, np.inf, np.pi, np.e, None

globals().update(UFUNCS)
globals().update(REDUCTIONS)


def get_array_module(*args):
    import sys
    return sys.modules[__name__] if builtins.any(isinstance(a, ndarray) for a in args) else np


def isscalar(x):
    return np.isscalar(x)


def shape(a):
    return a.shape


def ndim(a):
    return a.ndim


def size(a):
    return a.size


def reshape(a, shape):
    return asarray(a).reshape(shape)


def ravel(a):
    return asarray(a).ravel()


def transpose(a, axes=None):
    return asarray(a).transpose(axes)


def swapaxes(a, x, y):
    return asarray(a).swapaxes(x, y)


def squeeze(a, axis=None):
    return asarray(a).squeeze(axis)


def expand_dims(a, axis):
    a = asarray(a)
    axis = axis % (a.ndim + 1)
    return a.reshape(a.shape[:axis] + (1,) + a.shape[axis:])


def copy(a):
    return asarray(a).copy()


def flatnonzero(a):
    return asarray(np.flatnonzero(asnumpy(a)))


def nonzero(a):
    return where(a)


def take(a, indices, axis=None, out=None, mode="raise"):
    host = np.take(asnumpy(a), asnumpy(indices) if isinstance(indices, ndarray) else indices,
                   axis=axis, mode=mode)
    if out is not None:
        copyto(out, asarray(host))
        return out
    return asarray(host)


def roll(a, shift, axis=None):
    return asarray(np.roll(asnumpy(a), shift, axis=axis))


def tile(a, reps):
    return asarray(np.tile(asnumpy(a), reps))


def repeat(a, repeats, axis=None):
    return asarray(np.repeat(asnumpy(a), repeats, axis=axis))


def meshgrid(*xi, **kw):
    return [asarray(m) for m in np.meshgrid(*[asnumpy(x) for x in xi], **kw)]


def unique(a, **kw):
    r = np.unique(asnumpy(a), **kw)
    return tuple(asarray(x) for x in r) if isinstance(r, tuple) else asarray(r)


def sort(a, axis=-1):
    return asarray(np.sort(asnumpy(a), axis=axis))


def argsort(a, axis=-1, kind=None):
    # CuPy's signature: the MPAS longwave groups columns with kind='stable'.
    return asarray(np.argsort(asnumpy(a), axis=axis, kind=kind))


def searchsorted(a, v, side="left"):
    return asarray(np.searchsorted(asnumpy(a), asnumpy(v), side=side))


def cumsum(a, axis=None, dtype=None, out=None):
    r = asarray(np.cumsum(asnumpy(a), axis=axis, dtype=dtype))
    if out is not None:
        out[...] = r
        return out
    return r


def diff(a, n=1, axis=-1):
    return asarray(np.diff(asnumpy(a), n=n, axis=axis))


def dot(a, b):
    return asarray(np.dot(asnumpy(a), asnumpy(b)))


def array_equal(a, b, equal_nan=False):
    return bool(np.array_equal(asnumpy(a), asnumpy(b), equal_nan=equal_nan))


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return asarray(np.isclose(asnumpy(a), asnumpy(b), rtol=rtol, atol=atol, equal_nan=equal_nan))


def pad(a, pad_width, mode="constant", **kw):
    return asarray(np.pad(asnumpy(a), pad_width, mode=mode, **kw))


def atleast_1d(*arys):
    r = [asarray(a) if asarray(a).ndim else asarray(a).reshape(1) for a in arys]
    return r[0] if len(r) == 1 else r


# Host-evaluated array functions: NumPy on a host copy (unified memory, so
# no transfer over a bus), the result back in a facade array.  Word for word
# NumPy's -- index and integer work exactly as CuPy's; the floating ones
# (interp, gradient, cross, outer, matmul, linalg) are NumPy's arithmetic, a
# recorded delta against cuBLAS/cuSOLVER rather than a translated kernel.
def _on_host(np_fn):
    def fn(*args, **kwargs):
        def down(x):
            if isinstance(x, ndarray):
                return asnumpy(x)
            if isinstance(x, (list, tuple)):
                return type(x)(down(y) for y in x)
            return x

        def up(r):
            if isinstance(r, np.ndarray):
                return asarray(r)
            if isinstance(r, (list, tuple)):
                return type(r)(up(y) for y in r)
            return r
        return up(np_fn(*[down(a) for a in args], **{k: down(v) for k, v in kwargs.items()}))
    fn.__name__ = np_fn.__name__
    fn.__doc__ = f"NumPy's {np_fn.__name__} on the host (woof facade)."
    return fn


for _name in ("eye", "identity", "moveaxis", "broadcast_arrays", "bincount", "interp", "gradient", "cross",
              "diagonal", "isin", "lexsort", "unravel_index", "ravel_multi_index", "take_along_axis",
              "outer", "matmul", "nan_to_num", "packbits", "unpackbits", "fromfile", "load", "tril",
              "triu", "histogram", "digitize", "einsum", "trapz" if hasattr(np, "trapz") else "trapezoid"):
    globals()[_name] = _on_host(getattr(np, _name))
broadcast_shapes = np.broadcast_shapes


class _OGrid:
    def __getitem__(self, key):
        return [asarray(g) for g in np.ogrid[key]]


ogrid = _OGrid()
linalg = types.SimpleNamespace(**{n: _on_host(getattr(np.linalg, n))
                                  for n in ("eigh", "eigvalsh", "solve", "lstsq", "inv", "norm", "cholesky",
                                            "det", "svd", "qr")})


def putmask(a, mask, values):
    host = asnumpy(a)
    np.putmask(host, asnumpy(mask), asnumpy(values) if isinstance(values, ndarray) else values)
    copyto(a, asarray(host))


def shares_memory(a, b, max_work=None):
    """Same storage and overlapping byte intervals (a view shares its base's
    storage).  Interval overlap, as CuPy's may_share_memory answers it."""
    if not (isinstance(a, ndarray) and isinstance(b, ndarray)) or a._st is not b._st:
        return False
    if a.size == 0 or b.size == 0:
        return False

    def bounds(x):
        lo = x._offset + builtins.sum(builtins.min(0, (n - 1) * st) for n, st in zip(x.shape, x.strides))
        hi = x._offset + builtins.sum(builtins.max(0, (n - 1) * st) for n, st in zip(x.shape, x.strides)) + x.itemsize
        return lo, hi
    (alo, ahi), (blo, bhi) = bounds(a), bounds(b)
    return alo < bhi and blo < ahi


may_share_memory = shares_memory


def clear_memo():
    return None


# FFTs: NumPy's pocketfft on the host -- a recorded delta against cuFFT
# (DA perturbations, stochastic patterns), never a parity claim.
fft = types.SimpleNamespace(**{n: _on_host(getattr(np.fft, n)) for n in (
    "fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2", "irfft2", "fftn", "ifftn", "rfftn",
    "irfftn", "hfft", "ihfft")})
fft.fftfreq, fft.rfftfreq = _on_host(np.fft.fftfreq), _on_host(np.fft.rfftfreq)
fft.fftshift, fft.ifftshift = _on_host(np.fft.fftshift), _on_host(np.fft.ifftshift)
fft.config = types.SimpleNamespace(enable_nd_planning=False, get_plan_cache=lambda: None)

# cupy.testing: NumPy's assertions on host copies
def _testing_on_host(fn):
    def check(*args, **kwargs):
        return fn(*[asnumpy(a) if isinstance(a, ndarray) else a for a in args], **kwargs)
    check.__name__ = fn.__name__
    return check


testing = types.SimpleNamespace(**{n: _testing_on_host(getattr(np.testing, n)) for n in (
    "assert_array_equal", "assert_array_almost_equal", "assert_allclose", "assert_array_less",
    "assert_array_max_ulp", "assert_array_almost_equal_nulp", "assert_equal")})

get_default_memory_pool = cuda.get_default_memory_pool
get_default_pinned_memory_pool = cuda.get_default_pinned_memory_pool


# ---------------------------------------------------------------- raw kernels

class RawModule:
    """``cupy.RawModule(code=..., options=...)`` over the Metal translator."""

    def __init__(self, code=None, path=None, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, name_expressions=None, jitify=False):
        if code is None and path is not None:
            with open(path, encoding="utf-8") as f:
                code = f.read()
        if code is None:
            raise ValueError("RawModule needs code= or path=")
        from woof.backend.metal.loader import load_unit
        import hashlib
        name = "rawmodule-" + hashlib.sha256(code.encode()).hexdigest()[:12]
        self._module = load_unit(name, code, options=tuple(options or ()))
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
        self.attributes = {"local_size_bytes": 0}

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        return self._function(grid, block, args, shared_mem=shared_mem, stream=stream)

    def compile(self, log_stream=None):
        return None


class _Unsupported:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise BackendUnsupported(f"cupy.{self._name} is not implemented by woof's mlx backend")


from woof.backend.facade._elementwise import ElementwiseKernel, ReductionKernel  # noqa: E402
fuse = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]  # noqa: E731

cupyx = types.ModuleType("cupyx")
cupyx.__woof_facade__ = True


def _cupyx_getattr(name):
    raise BackendUnsupported(f"cupyx.{name} is not implemented by woof's mlx backend")


cupyx.__getattr__ = _cupyx_getattr
# unified memory: a pinned host array is a host array
cupyx.empty_pinned = lambda shape, dtype=np.float64, order="C": np.empty(shape, dtype=dtype, order=order)
cupyx.zeros_pinned = lambda shape, dtype=np.float64, order="C": np.zeros(shape, dtype=dtype, order=order)


def _scipy_on_host(module, name):
    def fn(*a, **k):
        import importlib
        return _on_host(getattr(importlib.import_module(f"scipy.{module}"), name))(*a, **k)
    return fn


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__path__ = []          # a package, so `import cupyx.scipy.special` resolves
    return m


cupyx.scipy = _module("cupyx.scipy",
    special=_module("cupyx.scipy.special", **{n: _scipy_on_host("special", n)
                                              for n in ("erf", "erfc", "gamma", "gammaln")}),
    ndimage=_module("cupyx.scipy.ndimage", **{n: _scipy_on_host("ndimage", n) for n in (
        "minimum_filter", "maximum_filter", "uniform_filter", "gaussian_filter", "label", "convolve")}))
cupyx.__path__ = []
#: the facade's own submodules, for `import cupyx.scipy.special` (backend._FacadeFinder)
SUBMODULES = {"cupyx.scipy": cupyx.scipy, "cupyx.scipy.special": cupyx.scipy.special,
              "cupyx.scipy.ndimage": cupyx.scipy.ndimage}


def __getattr__(name):
    raise BackendUnsupported(f"cupy.{name} is not implemented by woof's mlx backend")
