"""MPAS-Ocean v8.4.2: ``ocn_diagnostic_solve`` on the GPU.

:func:`diagnostic_solve` is ``ocn_diagnostic_solve`` (``mpas_ocn_diagnostics.F``)
for one block, in the Fortran's order and over each loop's own halo extent
(``nCellsArray(i)``, ``nEdgesArray(i)``, ``nVerticesArray(i)``):

1. ``layerThicknessEdge``, then the missing-neighbour sentinels;
2. ``ocn_relativeVorticity_circulation`` and ``relativeVorticityCell``;
3. divergence, kinetic energy and the vertical velocities (transport and GM
   bolus too), the tangential velocity, and the vertex kinetic energy when
   ``config_include_KE_vertex`` is on;
4. the normalized relative and planetary vorticity at vertices, edges and
   cells, and APVM when ``config_apvm_scale_factor`` is on;
5. density (with the in-situ expansion coefficients for 'Jacobian_from_TS'),
   potential density and displaced density through
   :mod:`~woof.hex.mpas_ocean.eos_v842`;
6. the Montgomery potential, or pressure, ``zMid``, ``zTop`` and ``ssh``;
7. the Brunt-Vaisala frequency and the Richardson number;
8. the surface values, KPP's surface-layer averages when
   ``config_use_cvmix_kpp`` is on, the attenuation coefficients, the land-ice
   mask ('pressure_only'), ``pressureAdjustedSSH`` and ``gradSSH``.

The kernels (``diagnostics_v842.cu``) are held word for word to the v8.4.2
routines, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4
and 8 (``tools/mpas_ocean_oracle/run_diagnostics.F90``,
``tests/test_mpas_ocean_diagnostics_parity.py``).

v8.4.2's own behaviours kept as written:

- the missing-neighbour sentinels are the single-precision ``-1e34`` widened;
- below ``maxLevelCell`` only ``vertVelocityTop`` is set to 0 before the
  upward sums; the transport and GM-bolus sums start from the words there;
- the ``normalVelocitySurfaceLayer`` copy runs over ``nCellsArray`` edges
  (``nEdges`` is set from ``nCellsArray`` above it) unless APVM ran;
- KPP weighs its partial layer by ``FRACTION`` (the significand) of the layer
  index, not its fractional part;
- APVM reads normal vorticity gradients below ``maxLevelEdgeTop`` that it
  never computed; MPAS keeps ``normalVelocity``, their factor, 0 there, and
  the port computes them as 0.

The state arrays are dicts of device arrays named as in the Fortran pools
(``mesh``, ``state``, ``tracers``, ``diagnostics``, ``forcing``), in the
Fortran layout with MPAS's missing-neighbour element: ``a(nVertLevels,
nCells+1)`` is C-order ``(nCells+1, nVertLevels)``. The dimension entries
(``nCellsArray`` ...) are host integers. A forcing field whose package is off
is ``None``.

Refused by name: the land-ice flux modes that compute fluxes ('standalone',
'coupled') and any mode v8.4.2 does not allocate the land-ice fields for.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import eos_v842
from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_diagnostics.F:90-870,1279-1308,1541-1830",
    "src/core_ocean/shared/mpas_ocn_diagnostics_routines.F:67-142",
) + eos_v842.SOURCE_FILES
MODULE_OPTIONS = COMPILE_OPTIONS
GRAVITY = 9.80616
LAND_ICE_MODES = ("off", "pressure_only")
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("diagnostics_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_diagnostics_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    if n > 0:
        _function(precision, name)(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def check_config(configs: dict) -> None:
    """Refuse, by name, what this port does not carry."""
    mode = configs.get("config_land_ice_flux_mode", "off")
    if mode not in LAND_ICE_MODES:
        raise ConfigurationRefusal(
            knob="config_land_ice_flux_mode", value=mode,
            reason="the land-ice fluxes ('standalone', 'coupled') are not ported, and v8.4.2 allocates no land-ice "
                   "fields for another mode",
            declaration="config_land_ice_flux_mode = 'off' | 'pressure_only'")
    eos_v842.eos_type(configs)
    pgrad = configs.get("config_pressure_gradient_type", "pressure_and_zmid")
    if pgrad not in ("pressure_and_zmid", "MontgomeryPotential", "MontgomeryPotential_and_density",
                     "Jacobian_from_density", "Jacobian_from_TS"):
        raise ConfigurationRefusal(
            knob="config_pressure_gradient_type", value=pgrad, reason="not a type v8.4.2 carries",
            declaration="config_pressure_gradient_type = 'pressure_and_zmid' | 'MontgomeryPotential' | "
                        "'MontgomeryPotential_and_density' | 'Jacobian_from_density' | 'Jacobian_from_TS'")


def diagnostic_solve(*, dt: float, mesh: dict, state: dict, tracers: dict, diagnostics: dict, forcing: dict,
                     configs: dict) -> None:
    """``ocn_diagnostic_solve`` for one block, in place on ``diagnostics``
    (and the sentinels in ``state``/``tracers``, the land-ice mask in ``forcing``)."""
    import cupy as cp
    check_config(configs)
    h = state["layerThickness"]
    precision = _precision_of(h)
    real = PRECISIONS[precision]
    nl = int(mesh["nVertLevels"])
    vd = int(mesh["vertexDegree"])
    cells, edges, vertices = ([int(x) for x in mesh[k]] for k in ("nCellsArray", "nEdgesArray", "nVerticesArray"))
    me = mesh["edgesOnCell"].shape[1]
    me2 = mesh["edgesOnEdge"].shape[1]
    u = state["normalVelocity"]
    act = tracers["activeTracers"]
    nt = act.shape[2]
    it, isal = int(tracers["index_temperature"]), int(tracers["index_salinity"])
    d, m, f = diagnostics, mesh, forcing
    i32 = np.int32
    n_cells, n_edges, n_vertices = cells[-1], edges[-1], vertices[-1]

    # layerThicknessEdge, then the -1e34 sentinels at nEdges+1 and nCells+1
    _launch(precision, "ocn_diag_layer_thickness_edge_v842", n_edges * nl, (
        d["layerThicknessEdge"], h, m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(n_edges), i32(nl)))
    _launch(precision, "ocn_diag_sentinels_v842", nl, (u, h, act, i32(n_edges), i32(n_cells), i32(nl), i32(nt),
                                                       i32(it), i32(isal)))
    _launch(precision, "ocn_diag_vorticity_circulation_v842", int(m["nVertices"]) * nl, (
        d["relativeVorticity"], d["circulation"], u, m["dcEdge"], m["areaTriangle"], m["edgesOnVertex"],
        m["edgeSignOnVertex"], m["maxLevelVertexBot"], i32(m["nVertices"]), i32(nl), i32(vd)))
    vertex_to_cell = (m["kiteAreasOnVertex"], m["areaCell"], m["nEdgesOnCell"], m["kiteIndexOnCell"],
                      m["verticesOnCell"], m["maxLevelCell"])
    _launch(precision, "ocn_diag_vertex_to_cell_v842", cells[0] * nl, (
        d["relativeVorticityCell"], d["relativeVorticity"], *vertex_to_cell, i32(cells[0]), i32(nl), i32(me),
        i32(vd), i32(0)))

    # divergence, kinetic energy and the vertical velocities
    div_hu = [cp.zeros((max(cells[1], 1), nl), dtype=real) for _ in range(3)]
    _launch(precision, "ocn_diag_divergence_v842", cells[1] * nl, (
        d["divergence"], d["kineticEnergyCell"], *div_hu, u, d["normalTransportVelocity"],
        d["normalGMBolusVelocity"], d["layerThicknessEdge"], m["dcEdge"], m["dvEdge"], m["areaCell"],
        m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"], m["maxLevelCell"], i32(cells[1]), i32(nl),
        i32(me)))
    _launch(precision, "ocn_diag_vertical_velocity_v842", cells[1], (
        d["vertVelocityTop"], d["vertTransportVelocityTop"], d["vertGMBolusVelocityTop"], *div_hu,
        m["maxLevelCell"], i32(cells[1]), i32(nl)))
    _launch(precision, "ocn_diag_tangential_velocity_v842", edges[1] * nl, (
        d["tangentialVelocity"], u, m["weightsOnEdge"], m["nEdgesOnEdge"], m["edgesOnEdge"], m["maxLevelEdgeTop"],
        i32(edges[1]), i32(nl), i32(me2)))
    if configs["config_include_KE_vertex"]:
        n_v1 = m["areaTriangle"].shape[0]
        ke_vertex = cp.zeros((n_v1, nl), dtype=real)
        ke_on_cells = cp.zeros((m["areaCell"].shape[0], nl), dtype=real)
        _launch(precision, "ocn_diag_kinetic_energy_vertex_v842", n_vertices * nl, (
            ke_vertex, u, m["dcEdge"], m["dvEdge"], m["areaTriangle"], m["edgesOnVertex"], i32(n_vertices),
            i32(nl), i32(vd)))
        _launch(precision, "ocn_diag_vertex_to_cell_v842", n_cells * nl, (
            ke_on_cells, ke_vertex, *vertex_to_cell, i32(n_cells), i32(nl), i32(me), i32(vd), i32(1)))
        _launch(precision, "ocn_diag_kinetic_energy_blend_v842", n_cells * nl, (
            d["kineticEnergyCell"], ke_on_cells, i32(n_cells * nl)))

    # the normalized vorticities, and APVM
    nrv_vertex = cp.zeros((m["areaTriangle"].shape[0], nl), dtype=real)
    npv_vertex = cp.zeros_like(nrv_vertex)
    _launch(precision, "ocn_diag_normalized_vorticity_vertex_v842", vertices[2] * nl, (
        nrv_vertex, npv_vertex, d["relativeVorticity"], m["fVertex"], h, m["kiteAreasOnVertex"], m["areaTriangle"],
        m["cellsOnVertex"], m["maxLevelVertexBot"], i32(vertices[2]), i32(nl), i32(vd)))
    _launch(precision, "ocn_diag_normalized_vorticity_edge_v842", edges[2] * nl, (
        d["normalizedRelativeVorticityEdge"], d["normalizedPlanetaryVorticityEdge"], nrv_vertex, npv_vertex,
        m["verticesOnEdge"], m["maxLevelEdgeBot"], i32(edges[2]), i32(nl)))
    _launch(precision, "ocn_diag_vertex_to_cell_v842", cells[1] * nl, (
        d["normalizedRelativeVorticityCell"], nrv_vertex, *vertex_to_cell, i32(cells[1]), i32(nl), i32(me),
        i32(vd), i32(0)))
    # v8.4.2 sets nEdges to the cell count here; APVM, when it runs, resets it
    surface_edges = n_cells
    apvm = real(configs["config_apvm_scale_factor"])
    if apvm > real(np.float32(1e-10)):
        surface_edges = edges[1]
        _launch(precision, "ocn_diag_apvm_v842", edges[1] * nl, (
            d["normalizedRelativeVorticityEdge"], d["normalizedRelativeVorticityCell"], nrv_vertex, u,
            d["tangentialVelocity"], m["dcEdge"], m["dvEdge"], m["cellsOnEdge"], m["verticesOnEdge"],
            m["maxLevelEdgeTop"], m["maxLevelEdgeBot"], i32(edges[1]), i32(nl), real(apvm * real(dt))))

    # the equation of state over nCellsArray(3)
    pgrad = configs["config_pressure_gradient_type"]
    eos_mesh = {"maxLevelCell": m["maxLevelCell"], "refBottomDepth": m["refBottomDepth"]}
    eos_args = dict(mesh=eos_mesh, active_tracers=act, tracers_surface_value=d["tracersSurfaceValue"],
                    index_temperature=it, index_salinity=isal, configs=configs, n_cells=cells[2])
    if pgrad == "Jacobian_from_TS":
        eos_v842.equation_of_state_density(density=d["density"], k_displaced=0, displacement_type="relative",
                                           thermal_expansion_coeff=d["inSituThermalExpansionCoeff"],
                                           saline_contraction_coeff=d["inSituSalineContractionCoeff"], **eos_args)
    else:
        eos_v842.equation_of_state_density(density=d["density"], k_displaced=0, displacement_type="relative",
                                           **eos_args)
    eos_v842.equation_of_state_density(density=d["potentialDensity"], k_displaced=1, displacement_type="absolute",
                                       **eos_args)
    eos_v842.equation_of_state_density(density=d["displacedDensity"], k_displaced=1, displacement_type="relative",
                                       **eos_args)

    # the Montgomery potential, or pressure and the layer heights
    with_land_ice = configs.get("config_land_ice_flux_mode", "off") == "pressure_only"
    with_frazil = bool(configs.get("config_use_frazil_ice_formation", False))
    if pgrad == "MontgomeryPotential":
        _launch(precision, "ocn_diag_montgomery_v842", n_cells, (
            d["montgomeryPotential"], d["density"], h, m["bottomDepth"], i32(n_cells), i32(nl)))
    else:
        none = f["seaIcePressure"]
        _launch(precision, "ocn_diag_pressure_zmid_v842", n_cells, (
            d["pressure"], d["zMid"], d["zTop"], state["ssh"], d["density"], h, m["bottomDepth"],
            f["atmosphericPressure"], f["seaIcePressure"], f["frazilSurfacePressure"] if with_frazil else none,
            f["landIcePressure"] if with_land_ice else none, m["maxLevelCell"], i32(n_cells), i32(nl),
            i32(with_frazil), i32(with_land_ice)))
    rho_sw = real(configs["config_density0"])
    coef = -(real(GRAVITY) / rho_sw)
    _launch(precision, "ocn_diag_stability_v842", cells[2], (
        d["BruntVaisalaFreqTop"], d["RiTopOfCell"], d["displacedDensity"], d["density"], d["zMid"], u,
        m["dcEdge"], m["dvEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["maxLevelCell"],
        i32(cells[2]), i32(nl), i32(me), real(coef)))
    _launch(precision, "ocn_diag_tracers_surface_value_v842", cells[2] * nt, (
        d["tracersSurfaceValue"], act, i32(cells[2]), i32(nl), i32(nt)))
    _launch(precision, "ocn_diag_surface_velocity_v842", surface_edges, (
        d["normalVelocitySurfaceLayer"], u, i32(surface_edges), i32(nl)))
    if configs["config_use_cvmix_kpp"]:
        depth = real(configs["config_cvmix_kpp_surface_layer_averaging"])
        _launch(precision, "ocn_diag_kpp_cells_v842", cells[0], (
            d["tracersSurfaceLayerValue"], d["indexSurfaceLayerDepth"], act, h, m["maxLevelCell"], i32(cells[0]),
            i32(nl), i32(nt), depth))
        _launch(precision, "ocn_diag_kpp_edges_v842", edges[0], (
            d["normalVelocitySurfaceLayer"], u, d["layerThicknessEdge"], m["maxLevelEdgeTop"], i32(edges[0]),
            i32(nl), depth))
    _launch(precision, "ocn_diag_attenuation_v842", cells[1], (
        d["surfaceFluxAttenuationCoefficient"], d["surfaceFluxAttenuationCoefficientRunoff"], i32(cells[1]),
        real(configs["config_flux_attenuation_coefficient"]),
        real(configs["config_flux_attenuation_coefficient_runoff"])))
    if with_land_ice:
        _launch(precision, "ocn_diag_land_ice_mask_v842", int(m["nCells"]), (
            f["landIceFraction"], f["landIceMask"], i32(m["nCells"])))
    _launch(precision, "ocn_diag_pressure_adjusted_ssh_v842", cells[1], (
        d["pressureAdjustedSSH"], state["ssh"], f["seaIcePressure"], f["landIceDraft"] if with_land_ice
        else f["seaIcePressure"], i32(cells[1]), i32(with_land_ice), real(real(GRAVITY) * rho_sw)))
    _launch(precision, "ocn_diag_grad_ssh_v842", edges[1], (
        d["gradSSH"], d["pressureAdjustedSSH"], m["dcEdge"], m["cellsOnEdge"], m["edgeMask"], i32(edges[1]),
        i32(nl)))
