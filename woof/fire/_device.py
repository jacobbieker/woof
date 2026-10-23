"""The fire-spread CUDA module: direct NVRTC, no FMA, no flush to zero.

``kernels/fire_spread.cu`` with ``glibc_flt32.cuh`` and ``glibc_flt64.cuh``
prepended is compiled once per device through ``compile_using_nvrtc``, as
``woof.core.terrain_drag`` does and for the reason it records: CuPy's
``RawModule`` appends ``-ftz=true`` after the caller's options, and NVRTC
honours the last one.  With ``-fmad=false --ftz=false`` every float and
double operation is the IEEE operation the NumPy reference performs, and
every ``exp``/``pow`` is glibc's own word (the two headers), so the device
results equal the host references raw word for raw word.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

#: No FMA contraction (gfortran contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.fire._device:fire_spread"
_SOURCES = ("glibc_flt32.cuh", "glibc_flt64.cuh", "fire_spread.cu")
BLOCK = 256


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).resolve().parents[1] / "core" / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "fire_spread.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


@lru_cache(maxsize=None)
def _function(device: int, name: str):
    return _module(device).get_function(name)


def kernel(name: str):
    """One entry point of the module on the current device, cached."""
    import cupy as cp

    return _function(cp.cuda.Device().id, name)


def launch(name: str, n: int, args) -> None:
    """Launch ``name`` over ``n`` cells, one thread each."""
    if n == 0:
        return
    grid = ((int(n) + BLOCK - 1) // BLOCK,)
    kernel(name)(grid, (BLOCK,), tuple(args))


def array_namespace(*values):
    """CuPy when any input is a CUDA array, else NumPy (no CuPy import)."""
    if any(hasattr(v, "__cuda_array_interface__") for v in values):
        import cupy as cp
        return cp
    return np


def device_key(xp):
    """The cache key for device-resident tables: the namespace and, for
    CuPy, the current device, so a second GPU gets its own copy."""
    if xp is np:
        return (xp, None)
    return (xp, xp.cuda.Device().id)


def device_array(xp, value, dtype, shape):
    """``value`` broadcast to ``shape``, C-contiguous, of ``dtype``."""
    out = xp.asarray(value, dtype=dtype)
    return xp.broadcast_to(out, shape).copy()
