"""MPAS-Seaice v8.4.2: the variational EVP internal stress on the GPU.

The kernels (``evp_variational_v842.cu``) transcribe

- ``mpas_seaice_velocity_solver_constitutive_relation.F``: ``seaice_init_evp``,
  ``seaice_evp_constitutive_relation`` and ``..._revised``;
- ``mpas_seaice_velocity_solver_variational.F``:
  ``seaice_internal_stress_variational`` (strain tensor, stress tensor and
  stress divergence) and ``seaice_final_divergence_shear_variational``.

Together they are the internal-stress step of the elastic subcycle when
``config_stress_divergence_scheme = 'variational'`` (the default). They are
held word for word to those routines, compiled verbatim with gfortran
``-O0 -ffp-contract=off``, in both of MPAS's precisions
(``tools/mpas_seaice_oracle``, ``tests/test_mpas_seaice_evp_parity.py``):

- ``r4``: RKIND = 4, MPAS's default build;
- ``r8``: RKIND = 8. The column package is double, so production MPAS-Seaice
  is built double. On Metal it runs through soft-f64.

Not covered here, and refused by name where a caller could reach them:

- the weak-form stress divergence (``config_stress_divergence_scheme = 'weak'``);
- the Wachspress and PWL basis setup (``seaice_init_velocity_solver_variational``),
  whose outputs (``basisGradientU/V``, ``basisIntegralsU/V/Metric``,
  ``tanLatVertexRotatedOverRadius``, ``cellVerticesAtVertex``) these kernels
  take as inputs;
- the rest of ``mpas_seaice_velocity_solver.F``: the velocity update and the
  ocean stress are :mod:`woof.hex.mpas_seaice.velocity_v842`, which composes
  them with these kernels into the elastic subcycle.

Arrays are device float32 (``r4``) or float64 (``r8``), int32 for
connectivity and masks, with the Fortran
layout: ``a(maxEdges, nCells)`` is C-order ``(nCells, maxEdges)``. See the
``.cu`` header.
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
    "src/core_seaice/shared/mpas_seaice_velocity_solver_constitutive_relation.F:28-49,64-294",
    "src/core_seaice/shared/mpas_seaice_velocity_solver_variational.F:123-257,271-651,665-806",
)
#: The unit's strict options. The arithmetic is explicit round-to-nearest
#: intrinsics, so the fast tuple (contraction on) computes the same words;
#: subnormals are kept in both modes (``keep_ftz``).
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
#: The census reads these (tools/metal_translation_census.py, composed units);
#: module_source() with no argument is the default-precision (r4) unit.
MODULE_OPTIONS = COMPILE_OPTIONS
PRECISIONS = {"r4": np.float32, "r8": np.float64}
STRESS_DIVERGENCE_SCHEMES = ("variational",)
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_seaice").joinpath("evp_variational_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


def require_scheme(config_stress_divergence_scheme: str) -> None:
    """Refuse a stress-divergence scheme this port does not carry, by name."""
    if config_stress_divergence_scheme not in STRESS_DIVERGENCE_SCHEMES:
        raise ConfigurationRefusal(
            knob="config_stress_divergence_scheme",
            value=config_stress_divergence_scheme,
            reason="only the variational stress divergence is ported to the GPU",
            declaration="config_stress_divergence_scheme = 'variational'",
        )


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    # RawModule appends -ftz=true after its caller options (CuPy 14.2); the
    # direct route keeps the explicit options, as the hex physics units do.
    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"seaice_evp_variational_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def _precision_of(array) -> str:
    for name, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return name
    raise ValueError(f"EVP arrays must be float32 (r4) or float64 (r8), not {array.dtype}")


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


#: the block's cell and vertex counts while :func:`block_counts` is in force:
#: the loops v8.4.2 bounds by nCells run to it over arrays that carry MPAS's
#: missing-cell element (whose whole-array operations still cover it)
_COUNTS = contextvars.ContextVar("mpas_seaice_block_counts", default=None)


@contextmanager
def block_counts(*, cells: int, vertices: int):
    """Run the kernels over arrays with MPAS's extra element: the loops the
    Fortran bounds by ``nCells``/``nVertices`` stop at them."""
    token = _COUNTS.set({"cells": int(cells), "vertices": int(vertices)})
    try:
        yield
    finally:
        _COUNTS.reset(token)


def loop_count(kind: str, n: int) -> int:
    """``nCells``/``nVertices`` within :func:`block_counts`, else ``n`` (the arrays' length)."""
    counts = _COUNTS.get()
    return n if counts is None else counts[kind]


def _grid(n: int) -> tuple[int]:
    return ((n + _BLOCK - 1) // _BLOCK,)


def init_evp(dv_edge, n_edges_solve: int, dynamics_time_step: float, elastic_time_step: float):
    """``seaice_init_evp`` for one block on one rank.

    Returns a device array of 3 words, of ``dvEdge``'s precision: ``dampingTimescale``,
    ``numericalInertiaCoefficient`` and ``evpDampingCriterion``. The model sets
    the last one, but no v8.4.2 routine reads it.
    """
    import cupy as cp
    if not 0 <= int(n_edges_solve) <= int(dv_edge.shape[0]):
        raise ValueError("n_edges_solve must lie within dvEdge")
    precision = _precision_of(dv_edge)
    dtype = PRECISIONS[precision]
    _check("dvEdge", dv_edge, dtype, (dv_edge.shape[0],))
    consts = cp.zeros(3, dtype=dtype)
    _function(precision, "seaice_init_evp_v842")((1,), (1,), (
        dv_edge, np.int32(n_edges_solve), dtype(dynamics_time_step),
        dtype(elastic_time_step), consts))
    return consts


def internal_stress_variational(*, mesh: dict, solver: dict, variational: dict, consts,
                                elastic_time_step: float, revised_evp: bool) -> None:
    """``seaice_internal_stress_variational`` for one block, updating in place.

    ``mesh``: ``nEdgesOnCell``, ``verticesOnCell``, ``cellsOnVertex``,
    ``areaTriangle`` and ``vertexDegree`` (the stress tensor's ``areaCell``
    argument is unused by the relation, so it is not taken).
    ``solver``: ``solveStress``, ``solveVelocity``, ``uVelocity``,
    ``vVelocity``, ``icePressure``, ``stressDivergenceU`` and
    ``stressDivergenceV``. ``variational``: ``strain11/22/12``,
    ``stress11/22/12``, ``replacementPressure``, ``cellVerticesAtVertex``,
    ``tanLatVertexRotatedOverRadius`` and the five ``basis*`` arrays.
    ``consts`` is :func:`init_evp`'s result.
    """
    import cupy as cp
    precision = _precision_of(solver["uVelocity"])
    dtype = PRECISIONS[precision]
    n_cells, max_edges = variational["strain11"].shape
    n_vertices = solver["uVelocity"].shape[0]
    vertex_degree = int(mesh["vertexDegree"])
    _check("nEdgesOnCell", mesh["nEdgesOnCell"], cp.int32, (n_cells,))
    _check("verticesOnCell", mesh["verticesOnCell"], cp.int32, (n_cells, max_edges))
    _check("cellsOnVertex", mesh["cellsOnVertex"], cp.int32, (n_vertices, vertex_degree))
    _check("areaTriangle", mesh["areaTriangle"], dtype, (n_vertices,))
    _check("cellVerticesAtVertex", variational["cellVerticesAtVertex"], cp.int32, (n_vertices, vertex_degree))
    for name in ("solveStress",):
        _check(name, solver[name], cp.int32, (n_cells,))
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    for name in ("uVelocity", "vVelocity", "stressDivergenceU", "stressDivergenceV"):
        _check(name, solver[name], dtype, (n_vertices,))
    _check("icePressure", solver["icePressure"], dtype, (n_cells,))
    _check("tanLatVertexRotatedOverRadius", variational["tanLatVertexRotatedOverRadius"], dtype,
           (n_vertices,))
    for name in ("strain11", "strain22", "strain12", "stress11", "stress22", "stress12", "replacementPressure"):
        _check(name, variational[name], dtype, (n_cells, max_edges))
    for name in ("basisGradientU", "basisGradientV", "basisIntegralsU", "basisIntegralsV",
                 "basisIntegralsMetric"):
        _check(name, variational[name], dtype, (n_cells, max_edges, max_edges))
    _check("consts", consts, dtype, (3,))
    v = variational
    _function(precision, "seaice_strain_tensor_variational_v842")(_grid(n_cells), (_BLOCK,), (
        v["strain11"], v["strain22"], v["strain12"], solver["uVelocity"], solver["vVelocity"],
        v["basisGradientU"], v["basisGradientV"], v["tanLatVertexRotatedOverRadius"],
        solver["solveStress"], mesh["nEdgesOnCell"], mesh["verticesOnCell"],
        np.int32(n_cells), np.int32(max_edges)))
    n_loop = loop_count("cells", n_cells)
    _function(precision, "seaice_stress_tensor_variational_v842")(_grid(n_loop), (_BLOCK,), (
        v["stress11"], v["stress22"], v["stress12"], v["strain11"], v["strain22"], v["strain12"],
        solver["icePressure"], v["replacementPressure"], solver["solveStress"], mesh["nEdgesOnCell"],
        consts, dtype(elastic_time_step), np.int32(bool(revised_evp)), np.int32(n_loop),
        np.int32(max_edges)))
    _function(precision, "seaice_stress_divergence_variational_v842")(_grid(n_vertices), (_BLOCK,), (
        solver["stressDivergenceU"], solver["stressDivergenceV"], v["stress11"], v["stress22"],
        v["stress12"], v["basisIntegralsU"], v["basisIntegralsV"], v["basisIntegralsMetric"],
        v["tanLatVertexRotatedOverRadius"], v["cellVerticesAtVertex"], solver["solveVelocity"],
        mesh["nEdgesOnCell"], mesh["cellsOnVertex"], mesh["areaTriangle"], np.int32(n_vertices),
        np.int32(vertex_degree), np.int32(max_edges)))


def final_divergence_shear_variational(*, mesh: dict, solver: dict, variational: dict,
                                       ridging: dict | None, use_column_package: bool) -> None:
    """``seaice_final_divergence_shear_variational`` for one block, in place.

    Writes ``solver["divergence"]`` and ``solver["shear"]`` (in %/day, the
    model's CICE comparison units). With ``use_column_package`` it also writes
    ``ridging["ridgeConvergence"]`` and ``ridging["ridgeShear"]``.
    """
    import cupy as cp
    precision = _precision_of(solver["divergence"])
    dtype = PRECISIONS[precision]
    n_cells, max_edges = variational["strain11"].shape
    for name in ("divergence", "shear"):
        _check(name, solver[name], dtype, (n_cells,))
    if use_column_package:
        if ridging is None:
            raise ValueError("use_column_package needs the ridging arrays")
        for name in ("ridgeConvergence", "ridgeShear"):
            _check(name, ridging[name], dtype, (n_cells,))
        ridge = (ridging["ridgeConvergence"], ridging["ridgeShear"])
    else:
        # the kernel does not touch them; any device pointer of the precision will do
        ridge = (solver["divergence"], solver["shear"])
    _function(precision, "seaice_final_divergence_shear_variational_v842")(_grid(n_cells), (_BLOCK,), (
        solver["divergence"], solver["shear"], *ridge, variational["strain11"], variational["strain22"],
        variational["strain12"], solver["solveStress"], mesh["nEdgesOnCell"],
        np.int32(bool(use_column_package)), np.int32(n_cells), np.int32(max_edges),
        np.int32(loop_count("cells", n_cells))))
