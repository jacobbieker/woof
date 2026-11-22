"""MPAS-Ocean v8.4.2: the Gent-McWilliams bolus velocity on the GPU.

:func:`compute_bolus_velocity` is ``ocn_gm_compute_Bolus_velocity``
(``mpas_ocn_gm.F``) for one block, in the Fortran's order and halo extents:

1. the opening fills (scratch fields to ``huge(0D0)``; the slopes, tapering,
   ``k33`` and bolus velocity to zero);
2. the vertical density derivative at the layer interfaces (``displacedDensity``
   of the layer above against ``density``, at most ``-1e-12``), averaged to
   edges; the along-layer density and ``zMid`` gradients and their values at
   the interfaces; the constant-z density gradient;
3. the relative isoneutral slope on edges (clipped to [-1, 1]), its RMS on
   cells, five vertical smoothing passes, the tapering (``config_max_relative_slope``,
   optionally the surface layer by ``boundaryLayerDepth`` and the bottom
   layer) and ``k33``, the tapering averaged to edges (``config_disable_redi_k33``
   zeroes ``k33``);
4. per edge, the stream function's tridiagonal boundary-value problem
   (``config_gravWaveSpeed_trunc``, ``config_standardGM_tracer_kappa``) and
   the bolus velocity, ``normalGMBolusVelocity``; the stream function on cells.

The kernels (``gm_v842.cu``) are held word for word to v8.4.2, compiled
verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_ocean_oracle/run_gm.F90``, ``tests/test_mpas_ocean_gm_parity.py``).
v8.4.2's own behaviour kept as written: at RKIND 4 the routine's
``huge(0D0)`` scratch fill is +Inf (gfortran compiles the file only with
``-fno-range-check``); on the outermost computed halo edges the slopes read
that fill from cells beyond ``nCellsArray(3)``.

Arrays are device float32 (RKIND 4) or float64 (RKIND 8) in the Fortran
layout with MPAS's missing-neighbour element: ``a(nVertLevels, nEdges+1)`` is
C-order ``(nEdges+1, nVertLevels)``, the ``TopOf`` fields carry
``nVertLevels+1`` levels.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = ("src/core_ocean/shared/mpas_ocn_gm.F",)
MODULE_OPTIONS = COMPILE_OPTIONS
GRAVITY = 9.80616
#: v8.4.2's Registry defaults for the GM and Redi configs
DEFAULT_CONFIGS = {
    "config_use_standardGM": False, "config_gravWaveSpeed_trunc": 0.3, "config_standardGM_tracer_kappa": 0.0,
    "config_max_relative_slope": 0.01, "config_Redi_kappa": 0.0, "config_disable_redi_k33": False,
    "config_use_Redi_surface_layer_tapering": False, "config_Redi_surface_layer_tapering_extent": 0.0,
    "config_use_Redi_bottom_layer_tapering": False, "config_Redi_bottom_layer_tapering_depth": 0.0,
}
#: the diagnostics the routine writes
OUTPUTS = ("normalGMBolusVelocity", "relativeSlopeTopOfEdge", "relativeSlopeTopOfCell", "relativeSlopeTapering",
           "relativeSlopeTaperingCell", "k33", "gmStreamFuncTopOfEdge", "gmStreamFuncTopOfCell")
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("gm_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_gm_v842_{precision}.cu")
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


def _config(configs: dict, name: str):
    return configs.get(name, DEFAULT_CONFIGS[name])


def compute_bolus_velocity(*, mesh: dict, diagnostics: dict, configs: dict) -> None:
    """``ocn_gm_compute_Bolus_velocity`` for one block, in place on the
    diagnostics' :data:`OUTPUTS` (it reads ``density``, ``displacedDensity``,
    ``zMid``, ``layerThicknessEdge``, ``BruntVaisalaFreqTop`` and, with the
    surface-layer tapering, ``boundaryLayerDepth``)."""
    import cupy as cp
    m, d = mesh, diagnostics
    precision = _precision_of(d["density"])
    real = PRECISIONS[precision]
    i32 = np.int32
    nl = int(m["nVertLevels"])
    cells = [int(x) for x in m["nCellsArray"]]
    edges = [int(x) for x in m["nEdgesArray"]]
    me = int(m["edgesOnCell"].shape[1])
    n_rows_c, n_rows_e = d["density"].shape[0], d["layerThicknessEdge"].shape[0]
    if cells[1] and int(cp.asnumpy(m["maxLevelCell"][: cells[1]]).min()) < 1:
        raise ValueError("v8.4.2's slope smoothing indexes maxLevelCell:nVertLevels: every cell of "
                         "nCellsArray(2) needs maxLevelCell >= 1")
    # mpas_allocate_scratch_field leaves the scratch fields as they were; the
    # oracle's stand-in (and these) start them at huge(1.0_RKIND)
    big = np.finfo(real).max
    grad_density_edge = cp.full((n_rows_e, nl), big, dtype=real)
    grad_zmid_edge = cp.full((n_rows_e, nl), big, dtype=real)
    grad_density_top = cp.full((n_rows_e, nl + 1), big, dtype=real)
    grad_zmid_top = cp.full((n_rows_e, nl + 1), big, dtype=real)
    grad_density_const_z = cp.full((n_rows_e, nl + 1), big, dtype=real)
    ddz_edge = cp.full((n_rows_e, nl + 1), big, dtype=real)
    ddz_cell = cp.full((n_rows_c, nl + 1), big, dtype=real)
    area_cell_sum = cp.full((n_rows_c, nl), big, dtype=real)

    _launch(precision, "ocn_gm_init_edges_v842", (edges[-1] + 1) * nl, (
        grad_density_edge, grad_density_top, ddz_edge, grad_zmid_edge, grad_zmid_top, d["relativeSlopeTopOfEdge"],
        d["relativeSlopeTapering"], d["normalGMBolusVelocity"], i32(edges[-1] + 1), i32(nl)))
    _launch(precision, "ocn_gm_init_cells_v842", (cells[-1] + 1) * nl, (
        ddz_cell, d["k33"], d["relativeSlopeTopOfCell"], d["relativeSlopeTaperingCell"], i32(cells[-1] + 1),
        i32(nl)))
    _launch(precision, "ocn_gm_ddensity_dz_cell_v842", cells[2], (
        ddz_cell, d["displacedDensity"], d["density"], d["zMid"], m["maxLevelCell"], i32(cells[2]), i32(nl)))
    _launch(precision, "ocn_gm_edge_slope_v842", edges[2], (
        ddz_edge, grad_density_edge, grad_zmid_edge, grad_density_top, grad_zmid_top, grad_density_const_z,
        d["relativeSlopeTopOfEdge"], ddz_cell, d["density"], d["zMid"], d["layerThicknessEdge"], m["dcEdge"],
        m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(edges[2]), i32(nl)))
    surface = bool(_config(configs, "config_use_Redi_surface_layer_tapering"))
    bld = d["boundaryLayerDepth"] if surface else d["density"]
    _launch(precision, "ocn_gm_cell_slope_v842", cells[-1] + 1, (
        d["relativeSlopeTopOfCell"], d["relativeSlopeTaperingCell"], d["k33"], area_cell_sum,
        d["relativeSlopeTopOfEdge"], m["dcEdge"], m["dvEdge"], d["zMid"], bld, m["nEdgesOnCell"], m["edgesOnCell"],
        m["maxLevelEdgeTop"], m["maxLevelCell"], i32(cells[-1] + 1), i32(cells[1]), i32(cells[-1]), i32(nl),
        i32(me), real(_config(configs, "config_max_relative_slope")), i32(surface),
        real(_config(configs, "config_Redi_surface_layer_tapering_extent")),
        i32(bool(_config(configs, "config_use_Redi_bottom_layer_tapering"))),
        real(_config(configs, "config_Redi_bottom_layer_tapering_depth")),
        i32(bool(_config(configs, "config_disable_redi_k33")))))
    _launch(precision, "ocn_gm_edge_tapering_v842", edges[2], (
        d["relativeSlopeTapering"], d["relativeSlopeTaperingCell"], m["cellsOnEdge"], m["maxLevelEdgeTop"],
        i32(edges[2]), i32(nl)))
    work = cp.zeros((max(edges[2], 1), 7 * nl), dtype=real)
    _launch(precision, "ocn_gm_stream_function_v842", edges[2], (
        d["gmStreamFuncTopOfEdge"], d["normalGMBolusVelocity"], work, d["BruntVaisalaFreqTop"],
        d["layerThicknessEdge"], grad_density_const_z, m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(edges[2]),
        i32(nl), real(_config(configs, "config_gravWaveSpeed_trunc")),
        real(_config(configs, "config_standardGM_tracer_kappa")), real(GRAVITY), real(configs["config_density0"])))
    _launch(precision, "ocn_gm_stream_function_cell_v842", cells[0], (
        d["gmStreamFuncTopOfCell"], d["gmStreamFuncTopOfEdge"], m["dcEdge"], m["dvEdge"], m["areaCell"],
        m["nEdgesOnCell"], m["edgesOnCell"], m["maxLevelEdgeTop"], i32(cells[0]), i32(nl), i32(me)))
