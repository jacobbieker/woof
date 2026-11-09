"""MPAS-Seaice v8.4.2: the EVP velocity solve on the GPU.

The kernels (``velocity_v842.cu``) transcribe the parts of
``mpas_seaice_velocity_solver.F`` that turn the internal stress into a
velocity:

- before the subcycle: ``coriolis_force_coefficient``, ``ocean_stress`` and
  ``init_subcycle_variables`` (variational);
- each elastic subcycle: ``ocean_stress_coefficient`` and ``solve_velocity``
  or ``solve_velocity_revised``;
- after it: ``principal_stresses_driver`` (variational).

:func:`subcycle` composes them with the internal stress of
:mod:`woof.hex.mpas_seaice.evp_v842` the way ``seaice_run_velocity_solver``
does, given the fields of the pre-subcycle routines not ported here (see
below). They are held word for word to those routines, compiled verbatim
with gfortran ``-O0 -ffp-contract=off``, in both of MPAS's precisions:

- ``r4``: RKIND = 4, MPAS's default build;
- ``r8``: RKIND = 8. The column package is double, so production MPAS-Seaice
  is built double.

See ``tools/mpas_seaice_oracle`` and
``tests/test_mpas_seaice_velocity_parity.py``.

The rest of the pre-subcycle work (``aggregate_mass_and_area``,
``calculation_masks``, ``new_ice_velocities``, ``ice_strength``, ``air_stress``,
``surface_tilt``) is :mod:`woof.hex.mpas_seaice.presubcycle_v842`; its outputs
are these kernels' inputs (``totalMassVertex``, ``iceAreaVertex``,
``solveStress``/``solveVelocity``, ``icePressure``, ``airStressVertexU/V``,
``surfaceTiltForceU/V``). Not ported yet: ``ocean_stress_final``. Halo
exchanges are a one-rank no-op.

Arrays are device float32 (``r4``) or float64 (``r8``), with int32 masks and
connectivity, in the Fortran layout: ``a(maxEdges, nCells)`` is C-order
``(nCells, maxEdges)``.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_velocity_solver.F:31-36,1664-1805,2149-2308,2326-2485,"
    "2499-2841,2859-3109",
)
#: Strict options. The arithmetic is explicit round-to-nearest intrinsics, so
#: the fast tuple computes the same words; subnormals are kept (``keep_ftz``).
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
#: The census reads these (tools/metal_translation_census.py, composed units);
#: module_source() with no argument is the default-precision (r4) unit.
MODULE_OPTIONS = COMPILE_OPTIONS
PRECISIONS = {"r4": np.float32, "r8": np.float64}
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_seaice").joinpath("velocity_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"seaice_velocity_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _precision_of(array) -> str:
    for name, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return name
    raise ValueError(f"velocity arrays must be float32 (r4) or float64 (r8), not {array.dtype}")


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


def _grid(n: int) -> tuple[int]:
    return ((max(n, 1) + _BLOCK - 1) // _BLOCK,)


def _vertex_fields(solver: dict, names, n_vertices: int, dtype) -> None:
    for name in names:
        _check(name, solver[name], dtype, (n_vertices,))


def _solve_bounds(n_vertices_solve: int, n_vertices: int) -> None:
    if not 0 <= int(n_vertices_solve) <= int(n_vertices):
        raise ValueError("nVerticesSolve must lie within nVertices")


def coriolis_and_ocean_stress(*, mesh: dict, icestate: dict, solver: dict, use_ocean_stress: bool) -> None:
    """``coriolis_force_coefficient`` then ``ocean_stress``, in place.

    ``mesh``: ``fVertex``, ``nVerticesSolve``. ``icestate``:
    ``totalMassVertex``. ``solver``: ``solveVelocity``,
    ``totalMassVertexfVertex``, ``oceanStressU/V``, ``uOceanVelocityVertex``,
    ``vOceanVelocityVertex``.
    """
    import cupy as cp
    precision = _precision_of(solver["uOceanVelocityVertex"])
    dtype = PRECISIONS[precision]
    n_vertices = solver["uOceanVelocityVertex"].shape[0]
    nvs = int(mesh["nVerticesSolve"])
    _solve_bounds(nvs, n_vertices)
    _vertex_fields(solver, ("totalMassVertexfVertex", "oceanStressU", "oceanStressV", "uOceanVelocityVertex",
                            "vOceanVelocityVertex"), n_vertices, dtype)
    _check("fVertex", mesh["fVertex"], dtype, (n_vertices,))
    _check("totalMassVertex", icestate["totalMassVertex"], dtype, (n_vertices,))
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    _function(precision, "seaice_coriolis_force_coefficient_v842")(_grid(nvs), (_BLOCK,), (
        solver["totalMassVertexfVertex"], icestate["totalMassVertex"], mesh["fVertex"], np.int32(nvs)))
    _function(precision, "seaice_ocean_stress_v842")(_grid(n_vertices), (_BLOCK,), (
        solver["oceanStressU"], solver["oceanStressV"], solver["uOceanVelocityVertex"],
        solver["vOceanVelocityVertex"], mesh["fVertex"], solver["solveVelocity"],
        np.int32(bool(use_ocean_stress)), np.int32(nvs), np.int32(n_vertices)))


def init_subcycle_variables(*, mesh: dict, solver: dict, variational: dict) -> None:
    """``init_subcycle_variables`` (variational), in place.

    ``mesh``: ``nVerticesSolve``. ``solver``: ``solveStress``,
    ``solveVelocity``, ``uVelocity``, ``vVelocity``, ``oceanStressCoeff``,
    ``stressDivergenceU/V``. ``variational``: ``strain11/22/12``,
    ``stress11/22/12``.
    """
    import cupy as cp
    precision = _precision_of(solver["uVelocity"])
    dtype = PRECISIONS[precision]
    n_vertices = solver["uVelocity"].shape[0]
    n_cells, max_edges = variational["strain11"].shape
    nvs = int(mesh["nVerticesSolve"])
    _solve_bounds(nvs, n_vertices)
    _vertex_fields(solver, ("uVelocity", "vVelocity", "oceanStressCoeff", "stressDivergenceU",
                            "stressDivergenceV"), n_vertices, dtype)
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    _check("solveStress", solver["solveStress"], cp.int32, (n_cells,))
    for name in ("strain11", "strain22", "strain12", "stress11", "stress22", "stress12"):
        _check(name, variational[name], dtype, (n_cells, max_edges))
    _function(precision, "seaice_init_subcycle_vertices_v842")(_grid(n_vertices), (_BLOCK,), (
        solver["stressDivergenceU"], solver["stressDivergenceV"], solver["uVelocity"], solver["vVelocity"],
        solver["oceanStressCoeff"], solver["solveVelocity"], np.int32(nvs), np.int32(n_vertices)))
    v = variational
    _function(precision, "seaice_init_subcycle_cells_variational_v842")(_grid(n_cells), (_BLOCK,), (
        v["strain11"], v["strain22"], v["strain12"], v["stress11"], v["stress22"], v["stress12"],
        solver["solveStress"], np.int32(n_cells), np.int32(max_edges)))


def pre_subcycle(*, mesh: dict, icestate: dict, solver: dict, variational: dict,
                 use_ocean_stress: bool) -> None:
    """``coriolis_force_coefficient``, ``ocean_stress`` and
    ``init_subcycle_variables`` (variational), in place, in the model's order
    (:func:`coriolis_and_ocean_stress`, then :func:`init_subcycle_variables`;
    ``presubcycle_v842.pre_subcycle`` puts the surface tilt between them).
    """
    coriolis_and_ocean_stress(mesh=mesh, icestate=icestate, solver=solver, use_ocean_stress=use_ocean_stress)
    init_subcycle_variables(mesh=mesh, solver=solver, variational=variational)


def velocity_update(*, mesh: dict, icestate: dict, solver: dict, inertia, use_ocean_stress: bool,
                    revised_evp: bool, elastic_time_step: float, dynamics_time_step: float) -> None:
    """One subcycle's ``ocean_stress_coefficient`` then ``solve_velocity``
    (or ``solve_velocity_revised``), in place.

    ``icestate``: ``totalMassVertex``, ``iceAreaVertex``. ``solver``
    additionally needs ``uVelocityInitial``, ``vVelocityInitial``,
    ``airStressVertexU/V`` and ``surfaceTiltForceU/V``. ``inertia`` is a
    one-word device array holding ``numericalInertiaCoefficient`` (the revised
    solve reads it), e.g. ``evp_v842.init_evp(...)[1:2]``.
    """
    import cupy as cp
    precision = _precision_of(solver["uVelocity"])
    dtype = PRECISIONS[precision]
    n_vertices = solver["uVelocity"].shape[0]
    nvs = int(mesh["nVerticesSolve"])
    _solve_bounds(nvs, n_vertices)
    _vertex_fields(solver, ("uVelocity", "vVelocity", "uVelocityInitial", "vVelocityInitial",
                            "totalMassVertexfVertex", "oceanStressU", "oceanStressV", "oceanStressCoeff",
                            "uOceanVelocityVertex", "vOceanVelocityVertex", "stressDivergenceU",
                            "stressDivergenceV", "airStressVertexU", "airStressVertexV",
                            "surfaceTiltForceU", "surfaceTiltForceV"), n_vertices, dtype)
    for name in ("totalMassVertex", "iceAreaVertex"):
        _check(name, icestate[name], dtype, (n_vertices,))
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    _check("inertia", inertia, dtype, (1,))
    s = solver
    _function(precision, "seaice_ocean_stress_coefficient_v842")(_grid(n_vertices), (_BLOCK,), (
        s["oceanStressCoeff"], s["uOceanVelocityVertex"], s["vOceanVelocityVertex"], s["uVelocity"],
        s["vVelocity"], icestate["iceAreaVertex"], s["solveVelocity"], np.int32(bool(use_ocean_stress)),
        np.int32(nvs), np.int32(n_vertices)))
    time_step = dtype(dynamics_time_step if revised_evp else elastic_time_step)
    _function(precision, "seaice_solve_velocity_v842")(_grid(nvs), (_BLOCK,), (
        s["uVelocity"], s["vVelocity"], s["uVelocityInitial"], s["vVelocityInitial"],
        icestate["totalMassVertex"], s["totalMassVertexfVertex"], s["stressDivergenceU"],
        s["stressDivergenceV"], s["airStressVertexU"], s["airStressVertexV"], s["surfaceTiltForceU"],
        s["surfaceTiltForceV"], s["oceanStressU"], s["oceanStressV"], s["oceanStressCoeff"],
        s["solveVelocity"], inertia, time_step, np.int32(bool(revised_evp)), np.int32(nvs)))


def principal_stresses(*, mesh: dict, variational: dict) -> None:
    """``principal_stresses_driver`` (variational), in place: writes
    ``variational["principalStress1/2"]`` over the ``nEdgesOnCell`` vertices of
    the first ``nCellsSolve`` cells (1e30 where the replacement pressure is
    not above puny)."""
    import cupy as cp
    precision = _precision_of(variational["stress11"])
    dtype = PRECISIONS[precision]
    n_cells, max_edges = variational["stress11"].shape
    ncs = int(mesh["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    _check("nEdgesOnCell", mesh["nEdgesOnCell"], cp.int32, (n_cells,))
    for name in ("stress11", "stress22", "stress12", "replacementPressure", "principalStress1",
                 "principalStress2"):
        _check(name, variational[name], dtype, (n_cells, max_edges))
    v = variational
    _function(precision, "seaice_principal_stresses_variational_v842")(_grid(ncs), (_BLOCK,), (
        v["principalStress1"], v["principalStress2"], v["stress11"], v["stress22"], v["stress12"],
        v["replacementPressure"], mesh["nEdgesOnCell"], np.int32(ncs), np.int32(max_edges)))


def subcycle(*, mesh: dict, icestate: dict, solver: dict, variational: dict, ridging: dict | None,
             n_subcycles: int, use_ocean_stress: bool, revised_evp: bool, use_column_package: bool,
             dynamics_time_step: float, elastic_time_step: float, stress_divergence_scheme: str = "variational"):
    """``seaice_run_velocity_solver`` from the internal stress on, at the
    precision of the arrays (``r4`` or ``r8``).

    ``seaice_init_evp``, the pre-subcycle routines ported here,
    ``n_subcycles`` elastic subcycles (internal stress, ocean stress
    coefficient, velocity solve), then ``final_divergence_shear`` and
    ``principal_stresses_driver``. The dicts carry the arrays the pieces
    name: :func:`pre_subcycle`, :func:`velocity_update`,
    :func:`principal_stresses` and ``evp_v842.internal_stress_variational``.
    Returns the device ``consts`` of ``evp_v842.init_evp``.
    """
    import cupy as cp
    from . import evp_v842 as evp

    evp.require_scheme(stress_divergence_scheme)
    consts = evp.init_evp(mesh["dvEdge"], int(mesh["nEdgesSolve"]), dynamics_time_step, elastic_time_step)
    pre_subcycle(mesh=mesh, icestate=icestate, solver=solver, variational=variational,
                 use_ocean_stress=use_ocean_stress)
    inertia = cp.ascontiguousarray(consts[1:2])
    for _ in range(int(n_subcycles)):
        evp.internal_stress_variational(mesh=mesh, solver=solver, variational=variational, consts=consts,
                                        elastic_time_step=elastic_time_step, revised_evp=revised_evp)
        velocity_update(mesh=mesh, icestate=icestate, solver=solver, inertia=inertia,
                        use_ocean_stress=use_ocean_stress, revised_evp=revised_evp,
                        elastic_time_step=elastic_time_step, dynamics_time_step=dynamics_time_step)
    evp.final_divergence_shear_variational(mesh=mesh, solver=solver, variational=variational, ridging=ridging,
                                           use_column_package=use_column_package)
    principal_stresses(mesh=mesh, variational=variational)
    return consts
