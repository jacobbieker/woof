"""MPAS-Ocean v8.4.2: four momentum/thickness tendency terms on the GPU.

The kernels (``tend_v842.cu``) transcribe ``ocn_vel_coriolis_tend``,
``ocn_vel_vadv_tend``, ``ocn_thick_hadv_tend`` and ``ocn_thick_vadv_tend``,
and the switches their ``_init`` routines set from the configs. Each adds
its term to a tendency array in place, as the Fortran does. They are held word
for word to those routines, compiled verbatim with gfortran
``-O0 -ffp-contract=off``, in both of MPAS's precisions:

- ``r4``: RKIND = 4, the default build;
- ``r8``: RKIND = 8, the precision MPAS-Ocean is run at.

See ``tools/mpas_ocean_oracle`` and ``tests/test_mpas_ocean_tend_parity.py``.

The time integrators that call these (``ocn_tend_vel``, ``ocn_tend_thick``,
split-explicit), and every other tendency term, are not ported yet;
docs/mpas-coupled-roadmap.md lists them.

Arrays are device float32 (``r4``) or float64 (``r8``), with int32
connectivity, in the Fortran layout: ``a(nVertLevels, nEdges)`` is C-order
``(nEdges, nVertLevels)``. See the ``.cu`` header.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_vel_coriolis.F:74-187,204-241",
    "src/core_ocean/shared/mpas_ocn_vel_vadv.F:74-168,183-213",
    "src/core_ocean/shared/mpas_ocn_thick_hadv.F:72-168,183-207",
    "src/core_ocean/shared/mpas_ocn_thick_vadv.F:72-147,162-186",
)
#: Strict options. The arithmetic is explicit round-to-nearest intrinsics, so
#: the fast tuple computes the same words; subnormals are kept (``keep_ftz``).
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
#: The census reads these (tools/metal_translation_census.py, composed units);
#: module_source() with no argument is the default-precision (r4) unit.
MODULE_OPTIONS = COMPILE_OPTIONS
PRECISIONS = {"r4": np.float32, "r8": np.float64}
#: config_time_integrator -> RK4On (ocn_vel_coriolis_init). Another value leaves
#: RK4On unset in the Fortran, so it is refused rather than guessed.
TIME_INTEGRATORS = {"RK4": 1, "split_explicit": 0, "unsplit_explicit": 0}
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("tend_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


def rk4_on(config_time_integrator: str) -> int:
    """``RK4On`` as ``ocn_vel_coriolis_init`` sets it, or a refusal by name."""
    try:
        return TIME_INTEGRATORS[config_time_integrator]
    except KeyError:
        raise ConfigurationRefusal(
            knob="config_time_integrator",
            value=config_time_integrator,
            reason="ocn_vel_coriolis_init sets RK4On only for these integrators",
            declaration="config_time_integrator = " + " | ".join(repr(k) for k in TIME_INTEGRATORS),
        ) from None


def vel_vadv_on(config_vert_coord_movement: str, config_disable_vel_vadv: bool) -> bool:
    """``velVadvOn`` as ``ocn_vel_vadv_init`` sets it."""
    return config_vert_coord_movement != "impermeable_interfaces" and not config_disable_vel_vadv


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_tend_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _precision_of(array) -> str:
    for name, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return name
    raise ValueError(f"tendency arrays must be float32 (r4) or float64 (r8), not {array.dtype}")


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


#: the block's halo extents (nCellsArray, nEdgesArray, nVerticesArray) while
#: :func:`halo_extents` is in force; otherwise every loop runs over its arrays
_EXTENTS = contextvars.ContextVar("mpas_ocean_halo_extents", default=None)


@contextmanager
def halo_extents(*, cells, edges, vertices):
    """Run the momentum and thickness terms over the Fortran's halo extents:
    each loop over ``nCellsArray(i)``/``nEdgesArray(i)``/``nVerticesArray(i)``
    as v8.4.2 writes it, the arrays the block's."""
    token = _EXTENTS.set({"cells": [int(x) for x in cells], "edges": [int(x) for x in edges],
                          "vertices": [int(x) for x in vertices]})
    try:
        yield
    finally:
        _EXTENTS.reset(token)


def loop_extent(kind: str, index: int, n: int) -> int:
    """The Fortran loop's extent (``n<Kind>Array(index)``) within :func:`halo_extents`, else ``n``."""
    extents = _EXTENTS.get()
    return n if extents is None else extents[kind][index - 1]


def _grid(n: int) -> tuple[int]:
    return ((n + _BLOCK - 1) // _BLOCK,)


def vel_coriolis_tend(*, mesh: dict, normalized_relative_vorticity_edge, normalized_planetary_vorticity_edge,
                      layer_thickness_edge, normal_velocity, kinetic_energy_cell, tend,
                      config_time_integrator: str, config_disable_vel_coriolis: bool = False) -> None:
    """``ocn_vel_coriolis_tend``: add the Coriolis and kinetic-energy gradient
    term to ``tend(nVertLevels, nEdges)``.

    ``mesh`` holds ``maxLevelEdgeTop``, ``nEdgesOnEdge``, ``cellsOnEdge``,
    ``edgesOnEdge``, ``weightsOnEdge``, ``dcEdge`` and ``edgeMask``.
    """
    import cupy as cp
    r4 = rk4_on(config_time_integrator)
    if config_disable_vel_coriolis:
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_edges, n_levels = tend.shape
    n_cells = kinetic_energy_cell.shape[0]
    max_edges2 = mesh["edgesOnEdge"].shape[1]
    for name, a in (("normalizedRelativeVorticityEdge", normalized_relative_vorticity_edge),
                    ("normalizedPlanetaryVorticityEdge", normalized_planetary_vorticity_edge),
                    ("layerThicknessEdge", layer_thickness_edge), ("normalVelocity", normal_velocity),
                    ("tend", tend)):
        _check(name, a, dtype, (n_edges, n_levels))
    _check("kineticEnergyCell", kinetic_energy_cell, dtype, (n_cells, n_levels))
    _check("maxLevelEdgeTop", mesh["maxLevelEdgeTop"], cp.int32, (n_edges,))
    _check("nEdgesOnEdge", mesh["nEdgesOnEdge"], cp.int32, (n_edges,))
    _check("cellsOnEdge", mesh["cellsOnEdge"], cp.int32, (n_edges, 2))
    _check("edgesOnEdge", mesh["edgesOnEdge"], cp.int32, (n_edges, max_edges2))
    _check("weightsOnEdge", mesh["weightsOnEdge"], dtype, (n_edges, max_edges2))
    _check("dcEdge", mesh["dcEdge"], dtype, (n_edges,))
    _check("edgeMask", mesh["edgeMask"], cp.int32, (n_edges, n_levels))
    n_edges_loop = loop_extent("edges", 1, n_edges)
    _function(precision, "ocn_vel_coriolis_tend_v842")(_grid(n_edges_loop), (_BLOCK,), (
        tend, normalized_relative_vorticity_edge, normalized_planetary_vorticity_edge, layer_thickness_edge,
        normal_velocity, kinetic_energy_cell, mesh["maxLevelEdgeTop"], mesh["nEdgesOnEdge"],
        mesh["cellsOnEdge"], mesh["edgesOnEdge"], mesh["weightsOnEdge"], mesh["dcEdge"], mesh["edgeMask"],
        np.int32(r4), np.int32(n_edges_loop), np.int32(n_levels), np.int32(max_edges2)))


def vel_vadv_tend(*, mesh: dict, normal_velocity, layer_thickness_edge, vert_ale_transport_top, tend,
                  config_vert_coord_movement: str, config_disable_vel_vadv: bool = False) -> None:
    """``ocn_vel_vadv_tend``: add the vertical momentum advection to
    ``tend(nVertLevels, nEdges)``. ``mesh`` holds ``maxLevelEdgeTop``,
    ``cellsOnEdge`` and ``edgeMask``."""
    import cupy as cp
    if not vel_vadv_on(config_vert_coord_movement, config_disable_vel_vadv):
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_edges, n_levels = tend.shape
    n_cells = vert_ale_transport_top.shape[0]
    _check("normalVelocity", normal_velocity, dtype, (n_edges, n_levels))
    _check("layerThicknessEdge", layer_thickness_edge, dtype, (n_edges, n_levels))
    _check("vertAleTransportTop", vert_ale_transport_top, dtype, (n_cells, n_levels + 1))
    _check("maxLevelEdgeTop", mesh["maxLevelEdgeTop"], cp.int32, (n_edges,))
    _check("cellsOnEdge", mesh["cellsOnEdge"], cp.int32, (n_edges, 2))
    _check("edgeMask", mesh["edgeMask"], cp.int32, (n_edges, n_levels))
    n_edges_loop = loop_extent("edges", 1, n_edges)
    _function(precision, "ocn_vel_vadv_tend_v842")(_grid(n_edges_loop), (_BLOCK,), (
        tend, normal_velocity, layer_thickness_edge, vert_ale_transport_top, mesh["maxLevelEdgeTop"],
        mesh["cellsOnEdge"], mesh["edgeMask"], np.int32(n_edges_loop), np.int32(n_levels)))


def thick_hadv_tend(*, mesh: dict, normal_velocity, layer_thickness_edge, tend,
                    config_disable_thick_hadv: bool = False) -> None:
    """``ocn_thick_hadv_tend``: add the horizontal thickness flux divergence to
    ``tend(nVertLevels, nCells)``. ``mesh`` holds ``maxLevelEdgeBot``,
    ``nEdgesOnCell``, ``edgesOnCell``, ``edgeSignOnCell``, ``dvEdge`` and
    ``areaCell``."""
    import cupy as cp
    if config_disable_thick_hadv:
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_cells, n_levels = tend.shape
    n_edges = normal_velocity.shape[0]
    max_edges = mesh["edgesOnCell"].shape[1]
    _check("normalVelocity", normal_velocity, dtype, (n_edges, n_levels))
    _check("layerThicknessEdge", layer_thickness_edge, dtype, (n_edges, n_levels))
    _check("maxLevelEdgeBot", mesh["maxLevelEdgeBot"], cp.int32, (n_edges,))
    _check("nEdgesOnCell", mesh["nEdgesOnCell"], cp.int32, (n_cells,))
    _check("edgesOnCell", mesh["edgesOnCell"], cp.int32, (n_cells, max_edges))
    _check("edgeSignOnCell", mesh["edgeSignOnCell"], cp.int32, (n_cells, max_edges))
    _check("dvEdge", mesh["dvEdge"], dtype, (n_edges,))
    _check("areaCell", mesh["areaCell"], dtype, (n_cells,))
    n_cells_loop = loop_extent("cells", 1, n_cells)
    _function(precision, "ocn_thick_hadv_tend_v842")(_grid(n_cells_loop), (_BLOCK,), (
        tend, normal_velocity, layer_thickness_edge, mesh["maxLevelEdgeBot"], mesh["nEdgesOnCell"],
        mesh["edgesOnCell"], mesh["edgeSignOnCell"], mesh["dvEdge"], mesh["areaCell"],
        np.int32(n_cells_loop), np.int32(n_levels), np.int32(max_edges)))


def thick_vadv_tend(*, mesh: dict, vert_ale_transport_top, tend, config_disable_thick_vadv: bool = False) -> None:
    """``ocn_thick_vadv_tend``: add the vertical transport divergence to
    ``tend(nVertLevels, nCells)``. ``mesh`` holds ``maxLevelCell``."""
    import cupy as cp
    if config_disable_thick_vadv:
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_cells, n_levels = tend.shape
    _check("vertAleTransportTop", vert_ale_transport_top, dtype, (n_cells, n_levels + 1))
    _check("maxLevelCell", mesh["maxLevelCell"], cp.int32, (n_cells,))
    n_cells_loop = loop_extent("cells", 1, n_cells)
    _function(precision, "ocn_thick_vadv_tend_v842")(_grid(n_cells_loop), (_BLOCK,), (
        tend, vert_ale_transport_top, mesh["maxLevelCell"], np.int32(n_cells_loop), np.int32(n_levels)))
