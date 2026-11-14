"""MPAS-Ocean v8.4.2: the pressure gradient and horizontal momentum mixing on the GPU.

The kernels (``pgrad_hmix_v842.cu``) transcribe:

- ``ocn_vel_pressure_grad_tend`` with every ``config_pressure_gradient_type``
  it accepts: 'pressure_and_zmid', 'MontgomeryPotential',
  'MontgomeryPotential_and_density', 'Jacobian_from_density' and
  'Jacobian_from_TS', the last two through ``pGrad_Jacobian_common_level``;
- ``ocn_vel_hmix_tend``: the Laplacian (``ocn_vel_hmix_del2_tend``), Leith
  (``ocn_vel_hmix_leith_tend``) and biharmonic (``ocn_vel_hmix_del4_tend``)
  forms, called in the Fortran's order, and the ``viscosity`` they add up.

Each adds into a tendency array in place, as the Fortran does. They are held
word for word to those routines, compiled verbatim with gfortran
``-O0 -ffp-contract=off``, at RKIND 4 and 8. See ``tools/mpas_ocean_oracle``
and ``tests/test_mpas_ocean_pgrad_hmix_parity.py``.

Not ported, and refused by name:

- the tensor forms of hmix (``config_use_mom_del2_tensor``,
  ``config_use_mom_del4_tensor``), which go through MPAS's
  ``mpas_tensor_operations``;
- any other pressure-gradient type (the Fortran logs it critical and stops).

``mpas_ocn_vel_pressure_grad.F`` also carries ``pGrad_Jacobian_POM_SCRUM``,
``pGrad_Jacobian_diagonal`` and ``pGrad_Jacobian_pseudo_flux``. They are
private to the module and no config reaches them in v8.4.2, so there is nothing
to port: a run of MPAS v8.4.2 cannot execute them.

Arrays are device float32 (``r4``) or float64 (``r8``), with int32
connectivity, in the Fortran layout of ``tend_v842``.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _check, _grid, _precision_of, loop_extent

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_vel_pressure_grad.F:80-411,625-664",
    "src/core_ocean/shared/mpas_ocn_vel_hmix_del2.F:81-194,385-415",
    "src/core_ocean/shared/mpas_ocn_vel_hmix_leith.F:84-204,219-243",
    "src/core_ocean/shared/mpas_ocn_vel_hmix_del4.F:82-279,506-534",
    "src/core_ocean/shared/mpas_ocn_vel_hmix.F:82-217",
)
MODULE_OPTIONS = COMPILE_OPTIONS
#: config_pressure_gradient_type -> the kernel's branch code
PGRAD_TYPES = {
    "pressure_and_zmid": 1,
    "MontgomeryPotential": 2,
    "MontgomeryPotential_and_density": 3,
    "Jacobian_from_density": 4,
    "Jacobian_from_TS": 5,
}
#: mpas_constants' gravity (a parameter, rounded to RKIND)
GRAVITY = 9.80616
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("pgrad_hmix_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


def pgrad_type_code(config_pressure_gradient_type: str) -> int:
    """The branch ``ocn_vel_pressure_grad_tend`` takes, or a refusal by name."""
    try:
        return PGRAD_TYPES[config_pressure_gradient_type]
    except KeyError:
        raise ConfigurationRefusal(
            knob="config_pressure_gradient_type",
            value=config_pressure_gradient_type,
            reason="ocn_vel_pressure_grad_tend logs any other type critical and stops",
            declaration="config_pressure_gradient_type = " + " | ".join(repr(k) for k in PGRAD_TYPES),
        ) from None


def pgrad_constants(config_density0: float, precision: str) -> tuple:
    """``density0Inv`` and ``gdensity0Inv`` as ``ocn_vel_pressure_grad_init``
    forms them in RKIND (rho_sw is config_density0, ocn_constants_init)."""
    dtype = PRECISIONS[precision]
    rho_sw = dtype(config_density0)
    return dtype(1.0) / rho_sw, dtype(GRAVITY) / rho_sw


def refuse_hmix_tensor(config_use_mom_del2_tensor: bool, config_use_mom_del4_tensor: bool) -> None:
    for knob, on in (("config_use_mom_del2_tensor", config_use_mom_del2_tensor),
                     ("config_use_mom_del4_tensor", config_use_mom_del4_tensor)):
        if on:
            raise ConfigurationRefusal(
                knob=knob, value=True,
                reason="the tensor hmix forms (mpas_tensor_operations) are not ported",
                declaration=f"{knob} = .false.")


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"ocn_pgrad_hmix_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def vel_pressure_grad_tend(*, mesh: dict, pressure, montgomery_potential, z_mid, density, potential_density,
                           tracers, index_temperature: int, index_salinity: int, tend,
                           config_pressure_gradient_type: str, config_common_level_weight: float,
                           config_density0: float, config_disable_vel_pgrad: bool = False,
                           in_situ_thermal_expansion_coeff=None, in_situ_saline_contraction_coeff=None) -> None:
    """``ocn_vel_pressure_grad_tend``: add the horizontal pressure gradient to
    ``tend(nVertLevels, nEdges)``.

    Cell fields are ``(nVertLevels, nCells)``; ``tracers`` is
    ``(nTracers, nVertLevels, nCells)`` and ``index_*`` are 1-based.
    ``mesh`` holds ``maxLevelEdgeTop``, ``cellsOnEdge``, ``dcEdge`` and
    ``edgeMask``. 'Jacobian_from_TS' needs the two in-situ coefficients.
    """
    import cupy as cp
    code = pgrad_type_code(config_pressure_gradient_type)
    if config_disable_vel_pgrad:
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_edges, n_levels = tend.shape
    n_cells = pressure.shape[0]
    n_tracers = tracers.shape[2]
    if code == PGRAD_TYPES["Jacobian_from_TS"]:
        if in_situ_thermal_expansion_coeff is None or in_situ_saline_contraction_coeff is None:
            raise ValueError("Jacobian_from_TS reads inSituThermalExpansionCoeff and inSituSalineContractionCoeff")
        if not (1 <= index_temperature <= n_tracers and 1 <= index_salinity <= n_tracers):
            raise ValueError(f"tracer indices must lie in 1..{n_tracers}")
    alpha = pressure if in_situ_thermal_expansion_coeff is None else in_situ_thermal_expansion_coeff
    beta = pressure if in_situ_saline_contraction_coeff is None else in_situ_saline_contraction_coeff
    for name, a in (("pressure", pressure), ("montgomeryPotential", montgomery_potential), ("zMid", z_mid),
                    ("density", density), ("potentialDensity", potential_density),
                    ("inSituThermalExpansionCoeff", alpha), ("inSituSalineContractionCoeff", beta)):
        _check(name, a, dtype, (n_cells, n_levels))
    _check("tracers", tracers, dtype, (n_cells, n_levels, n_tracers))
    _check("tend", tend, dtype, (n_edges, n_levels))
    _check("maxLevelEdgeTop", mesh["maxLevelEdgeTop"], cp.int32, (n_edges,))
    _check("cellsOnEdge", mesh["cellsOnEdge"], cp.int32, (n_edges, 2))
    _check("dcEdge", mesh["dcEdge"], dtype, (n_edges,))
    _check("edgeMask", mesh["edgeMask"], cp.int32, (n_edges, n_levels))
    density0_inv, gdensity0_inv = pgrad_constants(config_density0, precision)
    n_edges_loop = loop_extent("edges", 1, n_edges)
    _function(precision, "ocn_vel_pressure_grad_tend_v842")(_grid(n_edges_loop), (_BLOCK,), (
        tend, pressure, montgomery_potential, z_mid, density, potential_density, tracers, alpha, beta,
        mesh["maxLevelEdgeTop"], mesh["cellsOnEdge"], mesh["dcEdge"], mesh["edgeMask"], np.int32(code),
        density0_inv, gdensity0_inv, dtype(config_common_level_weight), np.int32(index_temperature),
        np.int32(index_salinity), np.int32(n_tracers), np.int32(n_edges_loop), np.int32(n_levels)))


def vel_hmix_tend(*, mesh: dict, divergence, relative_vorticity, viscosity, tend,
                  config_disable_vel_hmix: bool = False,
                  config_use_mom_del2: bool = True, config_mom_del2: float = 0.0,
                  config_use_Leith_del2: bool = False, config_Leith_parameter: float = 0.0,
                  config_Leith_dx: float = 0.0, config_Leith_visc2_max: float = 0.0,
                  config_use_mom_del4: bool = True, config_mom_del4: float = 0.0,
                  config_mom_del4_div_factor: float = 1.0,
                  config_use_mom_del2_tensor: bool = False, config_use_mom_del4_tensor: bool = False) -> None:
    """``ocn_vel_hmix_tend``: zero ``viscosity`` and add the del2, Leith and
    del4 momentum mixing to ``tend(nVertLevels, nEdges)`` in that order.

    The switches are the ``_init`` routines': del2 runs when
    ``config_use_mom_del2`` and ``config_mom_del2 > 0``, del4 likewise, Leith
    when ``config_use_Leith_del2``. ``divergence`` is ``(nVertLevels, nCells)``
    and ``relative_vorticity`` ``(nVertLevels, nVertices)``. ``mesh`` holds
    ``maxLevelEdgeTop``, ``maxLevelCell``, ``maxLevelVertexTop``,
    ``cellsOnEdge``, ``verticesOnEdge``, ``edgeMask``, ``dcEdge``, ``dvEdge``,
    ``meshScalingDel2``, ``meshScaling``, ``meshScalingDel4``; del4 also reads
    ``areaCell``, ``areaTriangle``, ``nEdgesOnCell``, ``edgesOnCell``,
    ``edgeSignOnCell``, ``edgesOnVertex`` and ``edgeSignOnVertex``.
    """
    import cupy as cp
    refuse_hmix_tensor(config_use_mom_del2_tensor, config_use_mom_del4_tensor)
    if config_disable_vel_hmix:
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_edges, n_levels = tend.shape
    n_cells = divergence.shape[0]
    n_vertices = relative_vorticity.shape[0]
    _check("tend", tend, dtype, (n_edges, n_levels))
    _check("viscosity", viscosity, dtype, (n_edges, n_levels))
    _check("divergence", divergence, dtype, (n_cells, n_levels))
    _check("relativeVorticity", relative_vorticity, dtype, (n_vertices, n_levels))
    _check("maxLevelEdgeTop", mesh["maxLevelEdgeTop"], cp.int32, (n_edges,))
    _check("cellsOnEdge", mesh["cellsOnEdge"], cp.int32, (n_edges, 2))
    _check("verticesOnEdge", mesh["verticesOnEdge"], cp.int32, (n_edges, 2))
    _check("edgeMask", mesh["edgeMask"], cp.int32, (n_edges, n_levels))
    for name in ("dcEdge", "dvEdge"):
        _check(name, mesh[name], dtype, (n_edges,))
    viscosity.fill(0)
    del2_on = config_use_mom_del2 and dtype(config_mom_del2) > 0
    del4_on = config_use_mom_del4 and dtype(config_mom_del4) > 0
    common = (mesh["maxLevelEdgeTop"], mesh["cellsOnEdge"], mesh["verticesOnEdge"])
    del2_leith = _function(precision, "ocn_vel_hmix_del2_leith_tend_v842")
    n_edges_loop = loop_extent("edges", 1, n_edges)
    if del2_on:
        _check("meshScalingDel2", mesh["meshScalingDel2"], dtype, (n_edges,))
        del2_leith(_grid(n_edges_loop), (_BLOCK,), (
            tend, viscosity, divergence, relative_vorticity, *common, mesh["meshScalingDel2"], mesh["dcEdge"],
            mesh["dvEdge"], mesh["edgeMask"], np.int32(0), dtype(config_mom_del2), dtype(0), dtype(0),
            np.int32(n_edges_loop), np.int32(n_levels)))
    if config_use_Leith_del2:
        _check("meshScaling", mesh["meshScaling"], dtype, (n_edges,))
        del2_leith(_grid(n_edges_loop), (_BLOCK,), (
            tend, viscosity, divergence, relative_vorticity, *common, mesh["meshScaling"], mesh["dcEdge"],
            mesh["dvEdge"], mesh["edgeMask"], np.int32(1), dtype(config_Leith_parameter), dtype(config_Leith_dx),
            dtype(config_Leith_visc2_max), np.int32(n_edges_loop), np.int32(n_levels)))
    if del4_on:
        max_edges = mesh["edgesOnCell"].shape[1]
        vertex_degree = mesh["edgesOnVertex"].shape[1]
        _check("meshScalingDel4", mesh["meshScalingDel4"], dtype, (n_edges,))
        _check("maxLevelCell", mesh["maxLevelCell"], cp.int32, (n_cells,))
        _check("maxLevelVertexTop", mesh["maxLevelVertexTop"], cp.int32, (n_vertices,))
        _check("nEdgesOnCell", mesh["nEdgesOnCell"], cp.int32, (n_cells,))
        _check("edgesOnCell", mesh["edgesOnCell"], cp.int32, (n_cells, max_edges))
        _check("edgeSignOnCell", mesh["edgeSignOnCell"], cp.int32, (n_cells, max_edges))
        _check("edgesOnVertex", mesh["edgesOnVertex"], cp.int32, (n_vertices, vertex_degree))
        _check("edgeSignOnVertex", mesh["edgeSignOnVertex"], cp.int32, (n_vertices, vertex_degree))
        _check("areaCell", mesh["areaCell"], dtype, (n_cells,))
        _check("areaTriangle", mesh["areaTriangle"], dtype, (n_vertices,))
        delsq_u = cp.empty((n_edges, n_levels), dtype)
        delsq_vorticity = cp.empty((n_vertices, n_levels), dtype)
        delsq_divergence = cp.empty((n_cells, n_levels), dtype)
        n_edges_loop = loop_extent("edges", 3, n_edges)
        _function(precision, "ocn_vel_hmix_del4_delsq_u_v842")(_grid(n_edges_loop), (_BLOCK,), (
            delsq_u, divergence, relative_vorticity, *common, mesh["dcEdge"], mesh["dvEdge"],
            np.int32(n_edges_loop), np.int32(n_levels)))
        n_vertices_loop = loop_extent("vertices", 2, n_vertices)
        _function(precision, "ocn_vel_hmix_del4_delsq_vorticity_v842")(_grid(n_vertices_loop), (_BLOCK,), (
            delsq_vorticity, delsq_u, mesh["maxLevelVertexTop"], mesh["edgesOnVertex"], mesh["edgeSignOnVertex"],
            mesh["dcEdge"], mesh["areaTriangle"], np.int32(n_vertices_loop), np.int32(n_levels),
            np.int32(vertex_degree)))
        n_cells_loop = loop_extent("cells", 2, n_cells)
        _function(precision, "ocn_vel_hmix_del4_delsq_divergence_v842")(_grid(n_cells_loop), (_BLOCK,), (
            delsq_divergence, delsq_u, mesh["maxLevelCell"], mesh["nEdgesOnCell"], mesh["edgesOnCell"],
            mesh["edgeSignOnCell"], mesh["dvEdge"], mesh["areaCell"], np.int32(n_cells_loop), np.int32(n_levels),
            np.int32(max_edges)))
        n_edges_loop = loop_extent("edges", 1, n_edges)
        _function(precision, "ocn_vel_hmix_del4_tend_v842")(_grid(n_edges_loop), (_BLOCK,), (
            tend, delsq_divergence, delsq_vorticity, *common, mesh["meshScalingDel4"], mesh["dcEdge"],
            mesh["dvEdge"], mesh["edgeMask"], dtype(config_mom_del4), dtype(config_mom_del4_div_factor),
            np.int32(n_edges_loop), np.int32(n_levels)))
