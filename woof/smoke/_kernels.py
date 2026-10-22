"""The smoke-physics NVRTC module: one composed unit, compiled directly.

Every device path in :mod:`woof.smoke` is held to WRF v4.7.1's own Fortran
bit for bit (``tools/smoke_wrf471_oracle``), so it is compiled the way the
terrain-drag module is (:mod:`woof.core.terrain_drag`): straight through
NVRTC with ``-fmad=false --ftz=false``, never through ``cupy.RawModule``
(which appends ``-ftz=true`` after the caller's options) and never as
CuPy array maths or an ``ElementwiseKernel`` (which contract products into
sums and call CUDA's own libm).  Fortran's real(4) ``EXP``, ``LOG`` and
``**`` are glibc's ``expf``, ``logf`` and ``powf`` (``glibc_flt32.cuh``:
``gfk_exp``, ``gfk_log``, ``gfk_pow``); the REAL*8 GOCART settling routine's
are glibc's ``exp`` and ``pow`` (``glibc_flt64.cuh``).

The unit is the two libm headers followed by the three smoke sources, in
this order (:data:`SOURCES`); none of the ``smoke_*.cu`` files compiles
alone.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

#: No FMA contraction (gfortran contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.smoke._kernels:smoke"
SOURCES = ("glibc_flt32.cuh", "glibc_flt64.cuh", "smoke_emission.cu",
           "smoke_plumerise.cu", "smoke_deposition.cu")
#: Threads per block.  Every kernel is one thread per output cell or per
#: column, so the launch shape cannot move a result.
BLOCK = 128


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).resolve().parents[1] / "core" / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "smoke.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def kernel(name: str):
    """One ``extern "C"`` kernel of the unit on the current device."""
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function(name)


def grid(n: int) -> tuple[int]:
    """Blocks for ``n`` threads at :data:`BLOCK` per block."""
    return ((int(n) + BLOCK - 1) // BLOCK,)
