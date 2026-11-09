"""MPAS-Seaice v8.4.2: the velocity solver's pre-subcycle routines on the GPU.

The kernels (``presubcycle_v842.cu``) transcribe the routines
``velocity_solver_pre_subcycle`` (``mpas_seaice_velocity_solver.F``) calls
before those :mod:`woof.hex.mpas_seaice.velocity_v842` already holds:

- ``aggregate_mass_and_area``;
- ``calculation_masks``: the initial ice area, the cell-to-vertex
  interpolation of the area and mass (``seaice_mesh``'s
  ``seaice_interpolate_cell_to_vertex``), ``stress_calculation_mask`` and
  ``velocity_calculation_mask``;
- ``new_ice_velocities``;
- ``ice_strength`` (the Hibler branch);
- ``air_stress`` with ``constant_air_stress``;
- ``surface_tilt``: geostrophic, sea-surface-height gradient, or none.

:func:`pre_subcycle` composes them with ``coriolis_force_coefficient``,
``ocean_stress`` and ``init_subcycle_variables`` in
``velocity_solver_pre_subcycle``'s order. Every word is held to those
routines compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at
RKIND 4 and RKIND 8 (``tools/mpas_seaice_oracle/run_presubcycle.F90``,
``tests/test_mpas_seaice_presubcycle_parity.py``). ``exp`` is the shared
glibc transcription: ``gfk_exp`` (``glibc_flt32.cuh``) at RKIND 4 and
``glibc_exp`` (``glibc_flt64.cuh``, MIT Arm optimized-routines) at RKIND 8.

Refused by name: ``ice_strength`` under the column package's vertical
thermodynamics (``config_use_column_package`` and
``config_use_column_vertical_thermodynamics``), which calls the column
package's ``colpkg_ice_strength`` (Rothrock 1975 by default, with the
ridging participation functions); that is not ported. Halo exchanges are a
one-rank no-op.

Arrays are device float32 (``r4``) or float64 (``r8``), with int32 masks and
1-based connectivity, in the Fortran layout: ``a(maxEdges, nCells)`` is
C-order ``(nCells, maxEdges)`` and ``a(1, nCategories, nCells)`` is
``(nCells, nCategories)``.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import velocity_v842
from .velocity_v842 import COMPILE_OPTIONS, PRECISIONS, _check, _grid, _precision_of, _solve_bounds

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_velocity_solver.F:31-36,546-1650,1819-2135",
    "src/core_seaice/shared/mpas_seaice_mesh.F:2742-2874",
)
MODULE_OPTIONS = COMPILE_OPTIONS
#: The shared libm transcription each precision's unit prepends
MATH_HEADERS = {"r4": ("glibc_flt32.cuh",), "r8": ("glibc_flt64.cuh",)}
_BLOCK = 128
SURFACE_TILTS = ("none", "geostrophic", "ssh_gradient")


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS[precision])
    body = files("woof.hex.mpas_seaice").joinpath("presubcycle_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + math + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"seaice_presubcycle_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def refuse_unported(*, use_column_package: bool, use_column_vertical_thermodynamics: bool) -> None:
    """Refuse, by name, the ice strength this port does not carry."""
    if use_column_package and use_column_vertical_thermodynamics:
        raise ConfigurationRefusal(
            knob="config_use_column_vertical_thermodynamics", value=True,
            reason="with the column package's vertical thermodynamics, ice_strength calls "
                   "colpkg_ice_strength (Rothrock 1975 and the ridging participation functions), "
                   "which is not ported",
            declaration="config_use_column_package = .false. or "
                        "config_use_column_vertical_thermodynamics = .false.")


def interpolate_cell_to_vertex(*, mesh: dict, vertex, cell) -> None:
    """``seaice_interpolate_cell_to_vertex``: ``vertex[:nVerticesSolve]`` is
    the ``areaCell``-weighted mean of ``cell`` over ``cellsOnVertex``; the
    halo vertices keep their words."""
    import cupy as cp
    precision = _precision_of(vertex)
    dtype = PRECISIONS[precision]
    n_vertices, n_cells = vertex.shape[0], cell.shape[0]
    vd = int(mesh["vertexDegree"])
    nvs = int(mesh["nVerticesSolve"])
    _solve_bounds(nvs, n_vertices)
    _check("variableCell", cell, dtype, (n_cells,))
    _check("areaCell", mesh["areaCell"], dtype, (n_cells,))
    _check("cellsOnVertex", mesh["cellsOnVertex"], cp.int32, (n_vertices, vd))
    _function(precision, "seaice_interpolate_cell_to_vertex_v842")(_grid(nvs), (_BLOCK,), (
        vertex, cell, mesh["areaCell"], mesh["cellsOnVertex"], np.int32(nvs), np.int32(vd)))


def _copy(precision: str, dst, src) -> None:
    kernel = "seaice_copy_int_v842" if dst.dtype == np.int32 else "seaice_copy_real_v842"
    _function(precision, kernel)(_grid(dst.size), (_BLOCK,), (dst, src, np.int32(dst.size)))


def _zero(precision: str, array) -> None:
    _function(precision, "seaice_zero_real_v842")(_grid(array.size), (_BLOCK,), (array, np.int32(array.size)))


def aggregate_and_masks(*, mesh: dict, boundary: dict, tracers: dict, aggregate: dict, icestate: dict,
                        ocean: dict, solver: dict, use_column_package: bool,
                        use_column_vertical_thermodynamics: bool) -> None:
    """``aggregate_mass_and_area`` then ``calculation_masks``, in place.

    ``tracers``: ``iceAreaCategory``, ``iceVolumeCategory``,
    ``snowVolumeCategory`` ``(nCells, nCategories)``. ``aggregate``:
    ``iceAreaCell``, ``iceVolumeCell``, ``snowVolumeCell``. ``icestate``:
    ``totalMassCell``, ``iceAreaCellInitial``, ``iceAreaVertex``,
    ``totalMassVertex``. ``mesh``: ``nEdgesOnCell``, ``cellsOnCell``,
    ``cellsOnVertex``, ``areaCell``, ``vertexDegree``, ``nVerticesSolve``.
    ``boundary``: ``interiorVertex``. ``ocean``: ``landIceMask``,
    ``landIceMaskVertex``. ``solver``: ``solveStress``, ``solveVelocity``.
    """
    import cupy as cp
    precision = _precision_of(aggregate["iceAreaCell"])
    dtype = PRECISIONS[precision]
    n_cells, n_cat = tracers["iceAreaCategory"].shape
    n_vertices = icestate["iceAreaVertex"].shape[0]
    max_edges = mesh["cellsOnCell"].shape[1]
    nvs = int(mesh["nVerticesSolve"])
    _solve_bounds(nvs, n_vertices)
    for name in ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory"):
        _check(name, tracers[name], dtype, (n_cells, n_cat))
    for name in ("iceAreaCell", "iceVolumeCell", "snowVolumeCell"):
        _check(name, aggregate[name], dtype, (n_cells,))
    for name in ("totalMassCell", "iceAreaCellInitial"):
        _check(name, icestate[name], dtype, (n_cells,))
    for name in ("iceAreaVertex", "totalMassVertex"):
        _check(name, icestate[name], dtype, (n_vertices,))
    _check("nEdgesOnCell", mesh["nEdgesOnCell"], cp.int32, (n_cells,))
    _check("cellsOnCell", mesh["cellsOnCell"], cp.int32, (n_cells, max_edges))
    _check("interiorVertex", boundary["interiorVertex"], cp.int32, (n_vertices,))
    _check("landIceMask", ocean["landIceMask"], cp.int32, (n_cells,))
    _check("landIceMaskVertex", ocean["landIceMaskVertex"], cp.int32, (n_vertices,))
    _check("solveStress", solver["solveStress"], cp.int32, (n_cells,))
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    a, i = aggregate, icestate
    _function(precision, "seaice_aggregate_mass_and_area_v842")(_grid(n_cells), (_BLOCK,), (
        a["iceAreaCell"], a["iceVolumeCell"], a["snowVolumeCell"], i["totalMassCell"],
        tracers["iceAreaCategory"], tracers["iceVolumeCategory"], tracers["snowVolumeCategory"],
        np.int32(n_cells), np.int32(n_cat)))
    if not (use_column_package and use_column_vertical_thermodynamics):
        _copy(precision, i["iceAreaCellInitial"], a["iceAreaCell"])
    interpolate_cell_to_vertex(mesh=mesh, vertex=i["iceAreaVertex"], cell=i["iceAreaCellInitial"])
    interpolate_cell_to_vertex(mesh=mesh, vertex=i["totalMassVertex"], cell=i["totalMassCell"])
    _function(precision, "seaice_stress_calculation_mask_v842")(_grid(n_cells), (_BLOCK,), (
        solver["solveStress"], i["iceAreaCellInitial"], i["totalMassCell"], ocean["landIceMask"],
        mesh["nEdgesOnCell"], mesh["cellsOnCell"], np.int32(n_cells), np.int32(max_edges)))
    _function(precision, "seaice_velocity_calculation_mask_v842")(_grid(n_vertices), (_BLOCK,), (
        solver["solveVelocity"], boundary["interiorVertex"], ocean["landIceMaskVertex"], i["iceAreaVertex"],
        i["totalMassVertex"], np.int32(nvs), np.int32(n_vertices)))


def new_ice_velocities(*, mesh: dict, ocean: dict, solver: dict) -> None:
    """``new_ice_velocities``, in place: the ocean velocity interpolated to
    the vertices, the velocities of newly solved vertices started from it,
    unsolved vertices zeroed, then ``solveVelocityPrevious`` and the initial
    velocities copied whole."""
    import cupy as cp
    s = solver
    precision = _precision_of(s["uVelocity"])
    dtype = PRECISIONS[precision]
    n_vertices = s["uVelocity"].shape[0]
    nvs = int(mesh["nVerticesSolve"])
    for name in ("uVelocity", "vVelocity", "uVelocityInitial", "vVelocityInitial", "stressDivergenceU",
                 "stressDivergenceV", "oceanStressU", "oceanStressV", "uOceanVelocityVertex",
                 "vOceanVelocityVertex"):
        _check(name, s[name], dtype, (n_vertices,))
    for name in ("solveVelocity", "solveVelocityPrevious"):
        _check(name, s[name], cp.int32, (n_vertices,))
    interpolate_cell_to_vertex(mesh=mesh, vertex=s["uOceanVelocityVertex"], cell=ocean["uOceanVelocity"])
    interpolate_cell_to_vertex(mesh=mesh, vertex=s["vOceanVelocityVertex"], cell=ocean["vOceanVelocity"])
    _function(precision, "seaice_new_ice_velocities_v842")(_grid(nvs), (_BLOCK,), (
        s["uVelocity"], s["vVelocity"], s["stressDivergenceU"], s["stressDivergenceV"], s["oceanStressU"],
        s["oceanStressV"], s["uOceanVelocityVertex"], s["vOceanVelocityVertex"], s["solveVelocity"],
        s["solveVelocityPrevious"], np.int32(nvs)))
    _copy(precision, s["solveVelocityPrevious"], s["solveVelocity"])
    _copy(precision, s["uVelocityInitial"], s["uVelocity"])
    _copy(precision, s["vVelocityInitial"], s["vVelocity"])


def ice_strength(*, mesh: dict, aggregate: dict, solver: dict, use_column_package: bool,
                 use_column_vertical_thermodynamics: bool) -> None:
    """``ice_strength``, in place: Hibler's ``icePressure`` over the first
    ``nCellsSolve`` cells (0 where the stress is not solved)."""
    import cupy as cp
    refuse_unported(use_column_package=use_column_package,
                    use_column_vertical_thermodynamics=use_column_vertical_thermodynamics)
    precision = _precision_of(solver["icePressure"])
    dtype = PRECISIONS[precision]
    n_cells = solver["icePressure"].shape[0]
    ncs = int(mesh["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    for name in ("iceVolumeCell", "iceAreaCell"):
        _check(name, aggregate[name], dtype, (n_cells,))
    _check("solveStress", solver["solveStress"], cp.int32, (n_cells,))
    _function(precision, "seaice_ice_strength_hibler_v842")(_grid(ncs), (_BLOCK,), (
        solver["icePressure"], aggregate["iceVolumeCell"], aggregate["iceAreaCell"], solver["solveStress"],
        np.int32(ncs)))


def air_stress(*, mesh: dict, atmos: dict, aggregate: dict, solver: dict, use_air_stress: bool,
               use_column_package: bool, use_column_vertical_thermodynamics: bool) -> None:
    """``air_stress``, in place: the constant-coefficient cell air stress
    (without the column vertical thermodynamics, which supplies its own),
    zeroed whole without ``config_use_air_stress``, then interpolated to the
    vertices."""
    s = solver
    precision = _precision_of(s["airStressCellU"])
    dtype = PRECISIONS[precision]
    n_cells = s["airStressCellU"].shape[0]
    ncs = int(mesh["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    for name in ("airStressCellU", "airStressCellV"):
        _check(name, s[name], dtype, (n_cells,))
    if not (use_column_package and use_column_vertical_thermodynamics):
        for name in ("uAirVelocity", "vAirVelocity", "airDensity"):
            _check(name, atmos[name], dtype, (n_cells,))
        _check("iceAreaCell", aggregate["iceAreaCell"], dtype, (n_cells,))
        _function(precision, "seaice_constant_air_stress_v842")(_grid(ncs), (_BLOCK,), (
            s["airStressCellU"], s["airStressCellV"], atmos["uAirVelocity"], atmos["vAirVelocity"],
            atmos["airDensity"], aggregate["iceAreaCell"], np.int32(ncs)))
    if not use_air_stress:
        _zero(precision, s["airStressCellU"])
        _zero(precision, s["airStressCellV"])
    interpolate_cell_to_vertex(mesh=mesh, vertex=s["airStressVertexU"], cell=s["airStressCellU"])
    interpolate_cell_to_vertex(mesh=mesh, vertex=s["airStressVertexV"], cell=s["airStressCellV"])


def surface_tilt(*, mesh: dict, icestate: dict, ocean: dict, solver: dict, tilt: str) -> None:
    """``surface_tilt``, in place: ``tilt`` is ``"none"``
    (``config_use_surface_tilt`` false: the force zeroed whole),
    ``"geostrophic"`` or ``"ssh_gradient"``
    (``config_geostrophic_surface_tilt`` true or false)."""
    import cupy as cp
    if tilt not in SURFACE_TILTS:
        raise ValueError(f"tilt must be one of {SURFACE_TILTS}")
    s = solver
    precision = _precision_of(s["surfaceTiltForceU"])
    dtype = PRECISIONS[precision]
    n_vertices = s["surfaceTiltForceU"].shape[0]
    nvs = int(mesh["nVerticesSolve"])
    _solve_bounds(nvs, n_vertices)
    if tilt == "none":
        _zero(precision, s["surfaceTiltForceU"])
        _zero(precision, s["surfaceTiltForceV"])
        return
    _check("solveVelocity", s["solveVelocity"], cp.int32, (n_vertices,))
    _check("totalMassVertex", icestate["totalMassVertex"], dtype, (n_vertices,))
    if tilt == "geostrophic":
        _check("fVertex", mesh["fVertex"], dtype, (n_vertices,))
        _function(precision, "seaice_surface_tilt_geostrophic_v842")(_grid(nvs), (_BLOCK,), (
            s["surfaceTiltForceU"], s["surfaceTiltForceV"], mesh["fVertex"], icestate["totalMassVertex"],
            s["uOceanVelocityVertex"], s["vOceanVelocityVertex"], s["solveVelocity"], np.int32(nvs)))
        return
    interpolate_cell_to_vertex(mesh=mesh, vertex=s["seaSurfaceTiltVertexU"], cell=ocean["seaSurfaceTiltU"])
    interpolate_cell_to_vertex(mesh=mesh, vertex=s["seaSurfaceTiltVertexV"], cell=ocean["seaSurfaceTiltV"])
    _function(precision, "seaice_surface_tilt_ssh_gradient_v842")(_grid(nvs), (_BLOCK,), (
        s["surfaceTiltForceU"], s["surfaceTiltForceV"], icestate["totalMassVertex"],
        s["seaSurfaceTiltVertexU"], s["seaSurfaceTiltVertexV"], s["solveVelocity"], np.int32(nvs)))


def pre_subcycle(*, mesh: dict, boundary: dict, tracers: dict, aggregate: dict, icestate: dict, ocean: dict,
                 atmos: dict, solver: dict, variational: dict, use_column_package: bool,
                 use_column_vertical_thermodynamics: bool, use_air_stress: bool, use_ocean_stress: bool,
                 tilt: str) -> None:
    """``velocity_solver_pre_subcycle``, in place, in the model's order:
    the aggregates and masks, the new-ice velocities, the ice strength, the
    air stress, then :func:`velocity_v842.pre_subcycle`'s Coriolis
    coefficient and ocean stress, the surface tilt, and the subcycle
    initialisation. The dicts carry the arrays the pieces name. Refuses the
    column package's ice strength by name before touching anything."""
    flags = {"use_column_package": use_column_package,
             "use_column_vertical_thermodynamics": use_column_vertical_thermodynamics}
    refuse_unported(**flags)
    if tilt not in SURFACE_TILTS:
        raise ValueError(f"tilt must be one of {SURFACE_TILTS}")
    aggregate_and_masks(mesh=mesh, boundary=boundary, tracers=tracers, aggregate=aggregate, icestate=icestate,
                        ocean=ocean, solver=solver, **flags)
    new_ice_velocities(mesh=mesh, ocean=ocean, solver=solver)
    ice_strength(mesh=mesh, aggregate=aggregate, solver=solver, **flags)
    air_stress(mesh=mesh, atmos=atmos, aggregate=aggregate, solver=solver, use_air_stress=use_air_stress, **flags)
    velocity_v842.coriolis_and_ocean_stress(mesh=mesh, icestate=icestate, solver=solver,
                                            use_ocean_stress=use_ocean_stress)
    surface_tilt(mesh=mesh, icestate=icestate, ocean=ocean, solver=solver, tilt=tilt)
    velocity_v842.init_subcycle_variables(mesh=mesh, solver=solver, variational=variational)
