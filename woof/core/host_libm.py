"""Binary64 transcendentals with glibc 2.39's words, element by element.

WRF's reference fixtures were computed by gfortran against glibc 2.39, so a
value graded against them -- static grid geometry, the projection
transcriptions, setup tables -- must be glibc 2.39's word on every machine.
NumPy's float64 ``exp``, ``log``, ``sin`` and friends call the C library on
most hosts, but NumPy 2.5 on an AVX-512 Linux machine runs its own vector
loops for several of them, and those round the last bit differently on a
few percent of arguments (scalars included: a 0-d call takes the same loop).
The C library itself is not glibc 2.39 everywhere either: macOS's libm
differs by an ULP on many arguments (128 ULP after the Lambert transforms
amplify it), and glibc 2.44 reduces huge arguments of ``sin``, ``cos`` and
``tan`` differently from 2.39.

These therefore take the CPU preprocessing bridge's host entries, which run
``tools/grib1_bridge/src/glibc239``: glibc 2.39's own ``exp``, ``log``,
``log10``, ``pow``, ``sin``, ``cos``, ``tan``, ``atan``, ``atan2``, ``asin``
and ``acos``, transcribed with the fused multiply-adds of the x86-64 FMA
build the references ran, so the words are glibc 2.39's on any host.  The
bridge splits the work over disjoint ranges on its explicit Rust worker
pool, which never changes an element, and applies the domain handling
below.  ``sqrt`` and ``x ** 2`` are not here: they are correctly rounded
everywhere.

Where the bridge is not installed (geometry is also used before
preparation's native assets are), every function falls back to :mod:`math`
one element at a time, which is the host C library: glibc 2.39's words only
on a glibc host whose functions match 2.39's.  An older installed bridge
holds host entries that call the C library; it serves the functions it has
and ``sin`` and ``cos`` fall back.  :func:`implementation` says which.
These are for initialization and setup, not per-step fields.
"""

from __future__ import annotations

import math
import ctypes
import os

import numpy as np

__all__ = ["exp", "log", "log10", "sin", "cos", "tan", "arctan", "arctan2",
           "arcsin", "arccos", "power", "scalar", "implementation",
           "GLIBC239", "C_LIBRARY"]

#: :func:`implementation` when the bridge's glibc 2.39 transcription runs.
GLIBC239 = "glibc-2.39-transcription"
#: :func:`implementation` when the host C library runs.
C_LIBRARY = "host-c-library"
#: The bridge export that marks the transcription (and its sin and cos).
_PORT_MARKER = "gpuwm_glibc239_sin"
#: The operation codes the transcription added to the host entries.
_PORT_ONLY = frozenset({"sin", "cos"})


def _library(pm):
    try:
        return pm._load()
    except FileNotFoundError:
        # Geometry/configuration callers also use this module before
        # preparation's native assets have been installed. Keep their
        # original scalar math path. Other load or ABI errors are real.
        return None


def _host_entry(pm, name, operation=None):
    library = _library(pm)
    if operation in _PORT_ONLY and not hasattr(library, _PORT_MARKER):
        return None  # an older bridge has no host sin or cos
    return getattr(library, name, None)


def implementation() -> str:
    """:data:`GLIBC239` where the installed bridge carries the glibc 2.39
    transcription, else :data:`C_LIBRARY`."""
    from woof.core import portable_math as pm
    return GLIBC239 if hasattr(_library(pm), _PORT_MARKER) else C_LIBRARY


def scalar(name: str):
    """The scalar C function ``name`` (``"exp"``, ``"pow"``, ``"expf"`` ...)
    with glibc 2.39's words: the bridge's transcription where it is
    installed, else the host C library's.  No domain handling: these are
    the C functions' values, for scalar references that transcribe C or
    Fortran calls one at a time.  The ``f`` functions are binary32."""
    from woof.core import portable_math as pm
    library = _library(pm)
    binary = name.startswith(("pow", "atan2"))
    kind = ctypes.c_float if name in ("expf", "logf", "powf") else ctypes.c_double
    if hasattr(library, _PORT_MARKER):
        function = getattr(library, "gpuwm_glibc239_" + name)
    else:
        process = ctypes.cdll.ucrtbase if os.name == "nt" else ctypes.CDLL(None)
        function = getattr(process, name)
    function.argtypes = [kind, kind] if binary else [kind]
    function.restype = kind
    return function


def _unary(function, values, name):
    from woof.core import portable_math as pm
    entry = _host_entry(pm, "gpuwm_host_unary_f64", name)
    array = np.require(values, dtype=np.float64, requirements=["C", "A"])
    if entry is not None:
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        entry.argtypes = [ctypes.c_uint32, pointer, pointer, size, size]
        entry.restype = ctypes.c_int32
        out = np.empty(array.shape, dtype=np.float64)
        code = entry(pm._UNARY_CODES[name], array.ctypes.data, out.ctypes.data,
                     array.size, pm._workers(None))
        if code:
            raise RuntimeError(f"native host {name} failed with code {code}")
        return out
    flat = np.fromiter((function(float(v)) for v in array.reshape(-1)),
                       dtype=np.float64, count=array.size)
    return flat.reshape(array.shape)


def _binary(function, left, right, name):
    from woof.core import portable_math as pm
    entry = _host_entry(pm, "gpuwm_host_binary_f64")
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape, b.shape)
    if entry is None:
        x, y = np.broadcast_arrays(a, b)
        flat = np.fromiter((function(float(u), float(v)) for u, v in
                            zip(x.reshape(-1), y.reshape(-1))),
                           dtype=np.float64, count=x.size)
        return flat.reshape(shape)
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [ctypes.c_uint32, pointer, size, pointer, size,
                     pointer, size, size]
    entry.restype = ctypes.c_int32
    out = np.empty(shape, dtype=np.float64)
    if not out.size:
        return out
    workers = pm._workers(None)

    def apply(x, y, target):
        x, y = (np.require(value, dtype=np.float64, requirements=["C", "A"])
                for value in (x, y))
        code = entry(pm._BINARY_CODES[name], x.ctypes.data, x.size,
                     y.ctypes.data, y.size, target.ctypes.data, target.size, workers)
        if code:
            raise RuntimeError(f"native host {name} failed with code {code}")

    if (a.size == 1 or (a.shape == shape and a.flags.c_contiguous)) and (
            b.size == 1 or (b.shape == shape and b.flags.c_contiguous)):
        apply(a, b, out)
    else:
        # General broadcasting marshals at most one bounded block per
        # operand, rather than allocating copies of a whole 3-D field.
        for x, y, target in np.nditer(
                [a, b, out], flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"], ["writeonly"]],
                order="C", buffersize=1 << 16):
            apply(x, y, target)
    return out


def _nan_outside(function, lower, upper):
    """``function`` with NumPy's NaN, not :mod:`math`'s ValueError, outside
    ``[lower, upper]``."""
    def guarded(value):
        if lower <= value <= upper:
            return function(value)
        return math.nan
    return guarded


def _log_domain(function):
    def guarded(value):
        if value > 0.0:
            return function(value)
        if value == 0.0:
            return -math.inf
        return math.nan
    return guarded


_LOG = _log_domain(math.log)
_LOG10 = _log_domain(math.log10)
_ASIN = _nan_outside(math.asin, -1.0, 1.0)
_ACOS = _nan_outside(math.acos, -1.0, 1.0)


def _exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _periodic(function):
    """``function`` with NumPy's NaN, not :mod:`math`'s ValueError, at an
    infinity."""
    def guarded(value):
        if math.isinf(value):
            return math.nan
        return function(value)
    return guarded


_SIN = _periodic(math.sin)
_COS = _periodic(math.cos)
_tan = _periodic(math.tan)


def exp(values) -> np.ndarray:
    return _unary(_exp, values, "exp")


def log(values) -> np.ndarray:
    return _unary(_LOG, values, "log")


def log10(values) -> np.ndarray:
    return _unary(_LOG10, values, "log10")


def sin(values) -> np.ndarray:
    return _unary(_SIN, values, "sin")


def cos(values) -> np.ndarray:
    return _unary(_COS, values, "cos")


def tan(values) -> np.ndarray:
    return _unary(_tan, values, "tan")


def arctan(values) -> np.ndarray:
    return _unary(math.atan, values, "atan")


def arctan2(y, x) -> np.ndarray:
    """glibc 2.39's ``atan2(y, x)``, over the broadcast operands.

    :func:`math.atan2` raises for no argument, and its answers for zeros,
    infinities and NaNs are the C library's, which are NumPy's.
    """
    return _binary(math.atan2, y, x, "atan2")


def arcsin(values) -> np.ndarray:
    return _unary(_ASIN, values, "asin")


def arccos(values) -> np.ndarray:
    return _unary(_ACOS, values, "acos")


def _pow(base, exponent):
    try:
        return math.pow(base, exponent)
    except OverflowError:
        odd = exponent.is_integer() and math.fmod(exponent, 2.0) != 0.0
        return -math.inf if base < 0.0 and odd else math.inf
    except ValueError:
        # math.pow refuses 0 ** negative (NumPy: inf) and a negative base
        # with a non-integer exponent (NumPy: nan).
        return math.inf if base == 0.0 else math.nan


def power(base, exponent) -> np.ndarray:
    """``base ** exponent`` through glibc 2.39's ``pow``.

    A scalar exponent of 2 or 0.5 is NumPy's square or square root, not
    ``pow``, in NumPy's own ``**``; callers that write those keep ``**``.
    """
    return _binary(_pow, base, exponent, "pow")
