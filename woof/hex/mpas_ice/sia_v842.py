"""MALI v8.4.2: the shallow-ice approximation (SIA) velocity solver on the GPU.

The kernels (``sia_v842.cu``) transcribe one call of ``li_sia_solve``:

- ``li_calculate_flowParamA``, for 'constant', 'PB1982' and 'CP2010';
- the upper surface interpolated to vertices, by barycentric weights or by
  kite areas;
- the tangential slope, by any of the three
  ``config_sia_tangent_slope_calculation`` methods, and the slope magnitude;
- the SIA velocity profile on every solved edge.

They are held word for word to those routines, compiled verbatim with gfortran
``-O0 -ffp-contract=off`` at RKIND 4 (MPAS's default single precision) and
RKIND 8 (``tools/mali_oracle``, ``tests/test_mali_sia_parity.py``). ``powf``
and ``expf`` come from the hex lane's glibc float32 transcription, and ``pow``
and ``exp`` from the binary64 one (``glibc_flt64.cuh``), both Arm
optimized-routines under MIT, which match the oracle host's libm (glibc 2.44).

Refused by name:

- ``config_velocity_solver`` other than 'sia'. 'FO', 'L1L2' and 'Stokes' are
  the external Albany/Trilinos dycores (``Interface_velocity_solver.cpp``),
  which have no pinned reference; 'simple' and 'none' are not ported.

Arrays are device float32 (RKIND 4) or float64 (RKIND 8), int32 for
connectivity and masks, in the Fortran
layout: ``a(nVertLevels, nCells)`` is C-order ``(nCells, nVertLevels)``.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import rkind

SOURCE_FILES = (
    "src/core_landice/mode_forward/mpas_li_sia.F:228-442",
    "src/core_landice/mode_forward/mpas_li_diagnostic_vars.F:381-509",
    "src/core_landice/shared/mpas_li_setup.F:484-536",
    "src/core_landice/shared/mpas_li_mask.F:738-757",
    "src/operators/mpas_geometry_utils.F:775-823",
    "src/operators/mpas_vector_operations.F:381-438",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
#: The census reads these (tools/metal_translation_census.py, composed units).
MODULE_OPTIONS = COMPILE_OPTIONS
VELOCITY_SOLVERS = ("sia",)
#: Albany/Trilinos external dycores: refused until a pinned reference exists.
EXTERNAL_DYCORES = ("FO", "L1L2", "Stokes")
TANGENT_SLOPE_CALCULATIONS = {"from_vertex_barycentric": 1, "from_vertex_barycentric_kiteareas": 2,
                              "from_normal_slope": 3}
FLOW_PARAM_A_CALCULATIONS = {"constant": 1, "PB1982": 2, "CP2010": 3}
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("sia_v842.cu").read_text(encoding="utf-8")
    if precision == "r8":
        return rkind.r8_prelude() + body
    from woof.hex.mpasphys.kessler_v842 import shared_math_header
    return shared_math_header() + body


def require_velocity_solver(config_velocity_solver: str) -> None:
    """Refuse a velocity solver this port does not carry, by name."""
    if config_velocity_solver in VELOCITY_SOLVERS:
        return
    if config_velocity_solver in EXTERNAL_DYCORES:
        reason = (f"'{config_velocity_solver}' is the external Albany/Trilinos dycore, which has no pinned "
                  "reference to hold a port to")
    else:
        reason = f"'{config_velocity_solver}' is not ported to the GPU"
    raise ConfigurationRefusal(knob="config_velocity_solver", value=config_velocity_solver, reason=reason,
                               declaration="config_velocity_solver = 'sia'")


def _option(table: dict, knob: str, value: str) -> int:
    try:
        return table[value]
    except KeyError:
        raise ConfigurationRefusal(knob=knob, value=value, reason="not an option li_sia_solve carries",
                                   declaration=f"{knob} = " + " | ".join(repr(k) for k in table)) from None


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"li_sia_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision,
                      _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def sia_solve(*, mesh: dict, geometry: dict, temperature, velocity: dict, n_edges_solve: int,
              config_ice_density: float, config_flowLawExponent: float, config_enhancementFactor: float,
              config_default_flowParamA: float, config_flowParamA_calculation: str,
              config_sia_tangent_slope_calculation: str, config_velocity_solver: str = "sia") -> None:
    """One ``li_sia_solve`` call for one block, in place.

    ``mesh``: ``cellsOnEdge``, ``verticesOnEdge``, ``baryCellsOnVertex``,
    ``baryWeightsOnVertex``, ``cellsOnVertex``, ``kiteAreasOnVertex``,
    ``nEdgesOnEdge``, ``edgesOnEdge``, ``weightsOnEdge``, ``dvEdge``,
    ``layerInterfaceSigma`` and ``layerCenterSigma``.
    ``geometry``: ``thickness``, ``upperSurface``, ``edgeMask``, ``cellMask``,
    ``normalSlopeEdge``, and the outputs ``slopeEdge``, ``tangentSlopeEdge``
    and ``upperSurfaceVertex``. ``temperature`` is (nCells, nVertLevels).
    ``velocity``: the outputs ``normalVelocity`` (nEdges, nVertInterfaces) and
    ``flowParamA`` (nCells, nVertLevels).
    """
    import cupy as cp
    require_velocity_solver(config_velocity_solver)
    tangent = _option(TANGENT_SLOPE_CALCULATIONS, "config_sia_tangent_slope_calculation",
                      config_sia_tangent_slope_calculation)
    flow = _option(FLOW_PARAM_A_CALCULATIONS, "config_flowParamA_calculation", config_flowParamA_calculation)
    precision = rkind.precision_of(temperature)
    real = rkind.PRECISIONS[precision]
    f32, i32 = real, cp.int32
    n_cells, n_levels = temperature.shape
    n_edges = mesh["dvEdge"].shape[0]
    n_vertices = mesh["baryCellsOnVertex"].shape[0]
    max_edges2 = mesh["edgesOnEdge"].shape[1]
    if not 0 <= int(n_edges_solve) <= n_edges:
        raise ValueError("n_edges_solve must lie within nEdges")
    for name, dtype, shape in (
            ("cellsOnEdge", i32, (n_edges, 2)), ("verticesOnEdge", i32, (n_edges, 2)),
            ("baryCellsOnVertex", i32, (n_vertices, 3)), ("baryWeightsOnVertex", f32, (n_vertices, 3)),
            ("cellsOnVertex", i32, (n_vertices, 3)), ("kiteAreasOnVertex", f32, (n_vertices, 3)),
            ("nEdgesOnEdge", i32, (n_edges,)), ("edgesOnEdge", i32, (n_edges, max_edges2)),
            ("weightsOnEdge", f32, (n_edges, max_edges2)), ("dvEdge", f32, (n_edges,)),
            ("layerInterfaceSigma", f32, (n_levels + 1,)), ("layerCenterSigma", f32, (n_levels,))):
        _check(name, mesh[name], dtype, shape)
    for name, dtype, shape in (
            ("thickness", f32, (n_cells,)), ("upperSurface", f32, (n_cells,)), ("edgeMask", i32, (n_edges,)),
            ("cellMask", i32, (n_cells,)), ("normalSlopeEdge", f32, (n_edges,)), ("slopeEdge", f32, (n_edges,)),
            ("tangentSlopeEdge", f32, (n_edges,)), ("upperSurfaceVertex", f32, (n_vertices,))):
        _check(name, geometry[name], dtype, shape)
    _check("temperature", temperature, f32, (n_cells, n_levels))
    _check("normalVelocity", velocity["normalVelocity"], f32, (n_edges, n_levels + 1))
    _check("flowParamA", velocity["flowParamA"], f32, (n_cells, n_levels))
    g = geometry
    _function(precision, "li_calculate_flowParamA_v842")(_grid(n_cells), (_BLOCK,), (
        velocity["flowParamA"], temperature, g["thickness"], mesh["layerCenterSigma"], np.int32(flow),
        real(config_default_flowParamA), real(config_enhancementFactor),
        real(config_ice_density), np.int32(n_cells), np.int32(n_levels)))
    if tangent in (1, 2):
        _function(precision, "li_upper_surface_vertex_v842")(_grid(n_vertices), (_BLOCK,), (
            g["upperSurfaceVertex"], g["upperSurface"], mesh["baryCellsOnVertex"], mesh["baryWeightsOnVertex"],
            mesh["cellsOnVertex"], mesh["kiteAreasOnVertex"], np.int32(tangent), np.int32(n_vertices)))
    _function(precision, "li_slope_edge_v842")(_grid(n_edges), (_BLOCK,), (
        g["tangentSlopeEdge"], g["slopeEdge"], g["normalSlopeEdge"], g["upperSurfaceVertex"],
        mesh["verticesOnEdge"], mesh["dvEdge"], g["edgeMask"], mesh["nEdgesOnEdge"], mesh["edgesOnEdge"],
        mesh["weightsOnEdge"], np.int32(tangent), np.int32(n_edges), np.int32(max_edges2)))
    _function(precision, "li_sia_velocity_v842")(_grid(n_edges_solve), (_BLOCK,), (
        velocity["normalVelocity"], velocity["flowParamA"], g["thickness"], g["slopeEdge"], g["normalSlopeEdge"],
        mesh["layerInterfaceSigma"], mesh["cellsOnEdge"], g["edgeMask"], g["cellMask"],
        real(config_ice_density), real(config_flowLawExponent), np.int32(n_edges_solve),
        np.int32(n_levels)))
