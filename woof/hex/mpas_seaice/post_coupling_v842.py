"""MPAS-Seaice v8.4.2: the sea ice's own work on the coupler's fields after an import.

:func:`post_atmospheric_coupling` and :func:`post_oceanic_coupling` are
v8.4.2's routines of those names (``mpas_seaice_forcing.F``), which a coupled
sea ice runs right after it takes the coupler's fields in:

- the atmospheric one, over ``nCellsSolve``: the wind speed from the air
  velocity, the air velocity rotated onto the sea ice's (rotated) grid
  (``seaice_latlon_vector_rotation_forward``), and the downward shortwave as
  the sum of its four bands;
- the oceanic one, over ``nCells`` (the halo too): the ocean velocity and the
  sea-surface tilt rotated the same way.

The rotation's sin, cos, asin and atan2 are the host libm's in MPAS; here
their words come in as ``trig`` ``(n, 8)``: per cell sin and cos of
``latCell``, ``lonCell`` and the rotated grid's latitude and longitude,
exactly what the E3SM cap's rotation_trig (woof.hex.coupled) makes
(the forward and backward rotations take the same eight words).

``real(8)`` only. The kernels (``post_coupling_v842.cu``) are held word for
word to v8.4.2's routines, compiled verbatim with gfortran ``-O0
-ffp-contract=off`` at RKIND 8, inside the hybrid sea ice's oracle
(``tools/e3sm_coupler_oracle/run_hybrid_seaice.F90``,
``tests/test_e3sm_hybrid_seaice_parity.py``). v8.4.2's own forcing streams,
which also call the atmospheric one, are not ported (``config_use_forcing``
is refused by :mod:`~woof.hex.mpas_seaice.timestep_v842`).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from .velocity_v842 import COMPILE_OPTIONS, launch_options

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_forcing.F:585-677",
    "src/core_seaice/shared/mpas_seaice_forcing.F:1503-1591",
    "src/core_seaice/shared/mpas_seaice_mesh.F:seaice_latlon_vector_rotation_forward",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128


def module_source() -> str:
    return files("woof.hex.mpas_seaice").joinpath("post_coupling_v842.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "seaice_post_coupling_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        _module_on(cp.cuda.Device().id, launch_options(COMPILE_OPTIONS)).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _f64(name: str, array, n: int):
    import cupy as cp
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a cupy (device) array; it is {type(array).__name__}")
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous float64 (real(8) only)")
    if array.ndim != 1 or int(array.shape[0]) < n:
        raise ValueError(f"{name} must be 1-D with at least {n} cells; it is {array.shape}")
    return array


def _trig(trig, n: int):
    import cupy as cp
    if not isinstance(trig, cp.ndarray):
        raise TypeError(f"trig must be a cupy (device) array; it is {type(trig).__name__}")
    if trig.dtype != np.float64 or not trig.flags.c_contiguous or trig.ndim != 2 or trig.shape[1] != 8 \
            or int(trig.shape[0]) < n:
        raise ValueError(f"trig must be C-contiguous float64 (n, 8) with n >= {n}; it is {trig.shape}")
    return trig


def post_atmospheric_coupling(*, pools: dict, configs: dict, trig) -> None:
    """``post_atmospheric_coupling`` in place: ``atmos_coupling``'s air
    velocity rotated, ``atmos_forcing``'s ``windSpeed`` and
    ``shortwaveDown``, over ``nCellsSolve``."""
    n = int(pools["mesh"]["nCellsSolve"])
    a, f = pools["atmos_coupling"], pools["atmos_forcing"]
    arrays = [_f64(f"atmos_coupling/{k}", a[k], n) for k in ("uAirVelocity", "vAirVelocity")]
    arrays += [_f64(f"atmos_forcing/{k}", f[k], n) for k in ("windSpeed", "shortwaveDown")]
    arrays += [_f64(f"atmos_coupling/{k}", a[k], n) for k in (
        "shortwaveVisibleDirectDown", "shortwaveVisibleDiffuseDown", "shortwaveIRDirectDown",
        "shortwaveIRDiffuseDown")]
    _launch("seaice_post_atmospheric_coupling_v842", n,
            (np.int32(n), *arrays, _trig(trig, n), np.int32(bool(configs["config_rotate_cartesian_grid"]))))


def post_oceanic_coupling(*, pools: dict, configs: dict, trig) -> None:
    """``post_oceanic_coupling`` in place: ``ocean_coupling``'s ocean
    velocity and sea-surface tilt rotated, over ``nCells``."""
    n = int(pools["mesh"]["nCells"])
    o = pools["ocean_coupling"]
    arrays = [_f64(f"ocean_coupling/{k}", o[k], n)
              for k in ("uOceanVelocity", "vOceanVelocity", "seaSurfaceTiltU", "seaSurfaceTiltV")]
    _launch("seaice_post_oceanic_coupling_v842", n,
            (np.int32(n), *arrays, _trig(trig, n), np.int32(bool(configs["config_rotate_cartesian_grid"]))))
