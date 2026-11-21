"""MPAS-Ocean v8.4.2: horizontal tracer mixing on the GPU.

:func:`tracer_hmix_tend` is ``ocn_tracer_hmix_tend`` for one tracer group on
one block: the Laplacian (``ocn_tracer_hmix_del2_tend``,
``config_use_tracer_del2`` with ``config_tracer_del2`` > 0) and then the
biharmonic (``ocn_tracer_hmix_del4_tend``, ``config_use_tracer_del4`` with
``config_tracer_del4`` > 0) mixing, then Redi isoneutral mixing
(``ocn_tracer_hmix_redi_tend``, on with ``config_use_standardGM``: its three
horizontal terms, each switched off by ``config_disable_redi_horizontal_term1/2/3``,
with ``config_Redi_kappa`` and GM's slopes and tapering), each added to
``tend`` in the Fortran's order over its own halo extent (``nCellsArray(1)``;
del4's first Laplacian over ``nCellsArray(2)``; Redi's vertical tracer
derivative over ``nCellsArray(2)``, its edge gradients over ``nEdgesArray(2)``).

The kernels (``tracer_hmix_v842.cu``) are held word for word to the v8.4.2
routines, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND
4 and 8 (``tools/mpas_ocean_oracle/run_tracer_hmix.F90``,
``tests/test_mpas_ocean_tracer_hmix_parity.py``).

Arrays are device float32 (RKIND 4) or float64 (RKIND 8) in the Fortran
layout with MPAS's missing-neighbour element: ``tracers(nTracers,
nVertLevels, nCells+1)`` is C-order ``(nCells+1, nVertLevels, nTracers)``.

v8.4.2's own behaviour kept as written: Redi's gradient at a column's top
interface is the along-layer gradient of level 2, so a one-level mesh raises
(the Fortran reads past the level). Redi's vertical term (``k33``) is
:mod:`~woof.hex.mpas_ocean.vmix_v842`'s.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_tracer_hmix.F",
    "src/core_ocean/shared/mpas_ocn_tracer_hmix_del2.F",
    "src/core_ocean/shared/mpas_ocn_tracer_hmix_del4.F",
    "src/core_ocean/shared/mpas_ocn_tracer_hmix_redi.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("tracer_hmix_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_tracer_hmix_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = launch_options()
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def check_config(configs: dict) -> None:
    """Refuse, by name, what this port does not carry (nothing since Redi is ported)."""


def switches(configs: dict) -> tuple[float | None, float | None]:
    """(eddyDiff2 or None, eddyDiff4 or None) as the _init routines set them."""
    del2 = float(configs["config_tracer_del2"]) if (configs.get("config_use_tracer_del2", False)
                                                    and configs["config_tracer_del2"] > 0.0) else None
    del4 = float(configs["config_tracer_del4"]) if (configs.get("config_use_tracer_del4", False)
                                                    and configs["config_tracer_del4"] > 0.0) else None
    return del2, del4


def redi_on(configs: dict) -> bool:
    """``rediOn`` as ``ocn_tracer_hmix_redi_init`` sets it."""
    return bool(configs.get("config_use_standardGM", False))


def tracer_hmix_tend(*, tracers, layer_thickness_edge, mesh: dict, tend, configs: dict,
                     diagnostics: dict | None = None) -> None:
    """``ocn_tracer_hmix_tend`` for one tracer group, in place on ``tend``.

    With Redi on, ``diagnostics`` carries ``zMid``, ``relativeSlopeTopOfEdge``
    and ``relativeSlopeTapering`` (``nVertLevels+1`` levels), as
    :func:`woof.hex.mpas_ocean.gm_v842.compute_bolus_velocity` leaves them."""
    import cupy as cp
    check_config(configs)
    if configs.get("config_disable_tr_hmix", False):
        return
    precision = _precision_of(tracers)
    real = PRECISIONS[precision]
    cells = [int(x) for x in mesh["nCellsArray"]]
    n_rows, nl, nt = tracers.shape
    me = mesh["edgesOnCell"].shape[1]
    m, i32 = mesh, np.int32
    del2, del4 = switches(configs)
    redi = redi_on(configs)
    if redi:  # checked before any tendency is written
        if diagnostics is None:
            raise ValueError("Redi (config_use_standardGM) needs the diagnostics' zMid and GM's slopes")
        if nl < 2:
            raise ValueError("v8.4.2's Redi reads the along-layer gradient of level 2: one-level meshes are not run")
    edges = (m["dvEdge"], m["dcEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"],
             m["cellsOnEdge"], m["maxLevelEdgeTop"])
    if del2 is not None:
        _launch(precision, "ocn_tracer_hmix_del2_v842", cells[0] * nl * nt, (
            tend, tracers, layer_thickness_edge, m["meshScalingDel2"], *edges, i32(cells[0]), i32(nl), i32(nt),
            i32(me), real(del2)))
    if del4 is not None:
        delsq = cp.zeros((n_rows, nl, nt), dtype=real)
        _launch(precision, "ocn_tracer_hmix_del4_laplacian_v842", cells[1] * nl * nt, (
            delsq, tracers, layer_thickness_edge, *edges, m["edgeMask"], i32(cells[1]), i32(nl), i32(nt), i32(me)))
        _launch(precision, "ocn_tracer_hmix_del4_v842", cells[0] * nl * nt, (
            tend, delsq, m["meshScalingDel4"], *edges, m["edgeMask"], i32(cells[0]), i32(nl), i32(nt), i32(me),
            real(del4)))
    if redi:
        _redi_tend(tracers=tracers, layer_thickness_edge=layer_thickness_edge, mesh=mesh, tend=tend,
                   configs=configs, diagnostics=diagnostics)


def _redi_tend(*, tracers, layer_thickness_edge, mesh: dict, tend, configs: dict, diagnostics: dict) -> None:
    """``ocn_tracer_hmix_redi_tend``: per-tracer scratch slices, zeroed as the
    Fortran's opening loops zero its scratch fields."""
    import cupy as cp
    precision = _precision_of(tracers)
    real = PRECISIONS[precision]
    m, d, i32 = mesh, diagnostics, np.int32
    cells = [int(x) for x in m["nCellsArray"]]
    edges = [int(x) for x in m["nEdgesArray"]]
    rows_c, nl, nt = tracers.shape
    rows_e = layer_thickness_edge.shape[0]
    me = m["edgesOnCell"].shape[1]
    kappa = real(configs.get("config_Redi_kappa", 0.0))
    dtdz_cell = cp.zeros((nt, rows_c, nl + 1), dtype=real)
    grad_edge = cp.zeros((nt, rows_e, nl), dtype=real)
    dtdz_edge = cp.zeros((nt, rows_e, nl + 1), dtype=real)
    grad_top_edge = cp.zeros((nt, rows_e, nl + 1), dtype=real)
    sloped = cp.zeros((nt, rows_c, nl + 1), dtype=real)
    area_sum = cp.zeros((nt, rows_c, nl), dtype=real)
    _launch(precision, "ocn_tracer_hmix_redi_dtdz_cell_v842", cells[1] * nt, (
        dtdz_cell, tracers, d["zMid"], m["maxLevelCell"], i32(cells[1]), i32(rows_c), i32(nl), i32(nt)))
    _launch(precision, "ocn_tracer_hmix_redi_edge_v842", edges[1] * nt, (
        grad_edge, dtdz_edge, grad_top_edge, dtdz_cell, tracers, layer_thickness_edge, m["dcEdge"],
        m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(edges[1]), i32(rows_c), i32(rows_e), i32(nl), i32(nt)))
    _launch(precision, "ocn_tracer_hmix_redi_sloped_cell_v842", cells[0] * nt, (
        sloped, area_sum, grad_top_edge, d["relativeSlopeTopOfEdge"], d["relativeSlopeTapering"], m["dcEdge"],
        m["dvEdge"], m["nEdgesOnCell"], m["edgesOnCell"], m["maxLevelEdgeTop"], m["maxLevelCell"], i32(cells[0]),
        i32(rows_c), i32(rows_e), i32(nl), i32(nt), i32(me)))
    _launch(precision, "ocn_tracer_hmix_redi_tend_v842", cells[0] * nl * nt, (
        tend, tracers, layer_thickness_edge, dtdz_edge, sloped, d["relativeSlopeTopOfEdge"],
        d["relativeSlopeTapering"], m["dvEdge"], m["dcEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"],
        m["edgeSignOnCell"], m["cellsOnEdge"], m["maxLevelEdgeTop"], m["maxLevelCell"], i32(cells[0]), i32(rows_c),
        i32(rows_e), i32(nl), i32(nt), i32(me), kappa,
        i32(not configs.get("config_disable_redi_horizontal_term1", False)),
        i32(not configs.get("config_disable_redi_horizontal_term2", False)),
        i32(not configs.get("config_disable_redi_horizontal_term3", False))))
