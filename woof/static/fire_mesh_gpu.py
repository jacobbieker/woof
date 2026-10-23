"""The fire-mesh kernels (``fire_mesh.cu``), compiled for WRF's words.

:mod:`woof.static.fuel` runs the fire-mesh refinement here when a GPU is
usable, and on its NumPy body otherwise.  Both evaluate WRF v4.7.1's REAL(4)
statements in WRF's order, so the two agree word for word with each other
and with WRF (tools/fire_mesh_wrf471_oracle,
tests/test_fire_mesh_wrf471_parity.py).

The module is compiled the way :mod:`woof.core.terrain_drag` compiles its
own, through NVRTC directly with ``-fmad=false --ftz=false``.
``cupy.RawModule`` appends ``-ftz=true`` after the caller's options, and
CuPy array maths fuses a product into a sum where the compiler chooses to;
either would move a bilinear weight off gfortran's word.

A static-preparation unit: it runs once, before any forecast state exists
on the device, so its source sits here and not among the forecast kernels
of ``woof/core/kernels`` that the VRAM frame pricing enumerates.  Its
kernels hold no per-thread arrays.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

#: No FMA contraction (gfortran -O0 contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.static.fire_mesh_gpu:fire_mesh"
SOURCE = Path(__file__).resolve().parent / "fire_mesh.cu"
_BLOCK = 256


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    return SOURCE.read_text(encoding="utf-8")


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
            source, MODULE_OPTIONS, None, "fire_mesh.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _launch(name: str, n: int, args: tuple) -> None:
    import cupy as cp

    kernel = _module(cp.cuda.Device().id).get_function(name)
    blocks = max(1, min((int(n) + _BLOCK - 1) // _BLOCK, 65535))
    kernel((blocks,), (_BLOCK,), args)


def _two_by_two(shape, what: str) -> None:
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise ValueError(
            f"{what} needs at least 2 x 2 points (WRF continues the field "
            f"from the two inside each edge), got shape {tuple(shape)}")


def refine_zsf(za_ext, i2, tx, j2, ty):
    """``fire_zsf``: the refined terrain, float32 ``(len(ty), len(tx))``.

    ``za_ext`` is the halo-extended atmospheric terrain, ``(ny+2, nx+2)``;
    ``i2``/``j2`` index its cells, so each must be in ``0 .. n`` of its axis.
    """
    import cupy as cp

    za = cp.ascontiguousarray(cp.asarray(za_ext, dtype=cp.float32))
    _two_by_two((za.shape[0] - 2, za.shape[1] - 2), "the fire-mesh terrain")
    i2 = cp.asarray(i2, dtype=cp.int32)
    j2 = cp.asarray(j2, dtype=cp.int32)
    tx = cp.asarray(tx, dtype=cp.float32)
    ty = cp.asarray(ty, dtype=cp.float32)
    if i2.shape != tx.shape or j2.shape != ty.shape:
        raise ValueError("refinement cells and weights differ in length")
    nxf, nyf = int(tx.size), int(ty.size)
    out = cp.empty((nyf, nxf), dtype=cp.float32)
    _launch("fire_zsf", nxf * nyf,
            (za, np.int32(za.shape[1]), i2, tx, j2, ty,
             np.int32(nxf), np.int32(nyf), out))
    return out


def slope(zsf, twodx, twody):
    """``fire_slope``: ``(dz/dx, dz/dy)`` of a float32 fire-mesh terrain."""
    import cupy as cp

    z = cp.ascontiguousarray(cp.asarray(zsf, dtype=cp.float32))
    _two_by_two(z.shape, "the fire-mesh slope")
    nyf, nxf = z.shape
    dzdx = cp.empty_like(z)
    dzdy = cp.empty_like(z)
    _launch("fire_slope", nxf * nyf,
            (z, np.int32(nxf), np.int32(nyf), np.float32(twodx),
             np.float32(twody), dzdx, dzdy))
    return dzdx, dzdy


def repeat(atm, sr_x: int, sr_y: int):
    """``fire_repeat``: uint8 categories, each parent cell's on its mesh."""
    import cupy as cp

    a = cp.ascontiguousarray(cp.asarray(atm, dtype=cp.uint8))
    ny, nx = a.shape
    nxf, nyf = nx * int(sr_x), ny * int(sr_y)
    out = cp.empty((nyf, nxf), dtype=cp.uint8)
    _launch("fire_repeat", nxf * nyf,
            (a, np.int32(nx), np.int32(sr_x), np.int32(sr_y),
             np.int32(nxf), np.int32(nyf), out))
    return out


def gather(block, row, col, missing):
    """``fire_gather``: int32 pixel codes, ``missing`` where ``row < 0``.

    ``row``/``col`` index ``block`` (the window the host read); the host
    computed them, so they are in range wherever ``row >= 0``.
    """
    import cupy as cp

    b = cp.ascontiguousarray(cp.asarray(block, dtype=cp.int32))
    r = cp.ascontiguousarray(cp.asarray(row, dtype=cp.int32))
    c = cp.ascontiguousarray(cp.asarray(col, dtype=cp.int32))
    if r.shape != c.shape:
        raise ValueError("pixel rows and columns differ in shape")
    out = cp.empty(r.shape, dtype=cp.int32)
    n = int(r.size)
    _launch("fire_gather", n,
            (b, np.int32(b.shape[1]), r, c, np.int64(n), np.int32(missing),
             out))
    return out
