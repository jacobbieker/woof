"""MPAS-Seaice v8.4.2's column ridging on the GPU.

``column_ridging`` (``mpas_seaice_column.F:2498-2767``) and the column package
under it (``colpkg_step_ridge``: ``ridge_ice``'s participation and
redistribution functions and its ridging iterations, then ``cleanup_itd``),
word for word with the gfortran -O0 -ffp-contract=off oracle at RKIND 8. One
thread is one owned cell; ``column_ridge_v842.cu`` is the transcription, on
``column_itd_v842.cu``'s ``cleanup_itd``. ``exp`` is ``glibc_exp`` (the MIT
transcription of glibc's binary64 exp), so nothing here needs an opt-in.

What it carries: both participation functions (``Thorndike75``,
``exponential``) and both redistribution functions (``Hibler80``,
``exponential``), any ``config_ridiging_efolding_scale``, dynamics subcycles
(``config_dynamics_subcycle_number``: ``dynamicsTimeStep`` is the ridging
step, its product with the subcycles the cleanup's), the topo ponds' water
lost with the ridging ice, and everything ``cleanup_itd`` carries
(:mod:`~woof.hex.mpas_seaice.column_itd_v842`).

What it refuses, by name: what the ITD thermodynamics port refuses (the
zero-layer thermodynamics, the aerosols, more than ``ITD_MAXCAT``
categories, the biogeochemistry but the brine height and v8.4.2's inert
vertical salinity, and the solved vertical salinity ``zsal_solve``). The
brine fraction is a volume tracer: new ridges take the ridging ice's.

v8.4.2's quirks are kept: ``ridge_prep``'s ``hin_max(ncat) = 1e8`` lands in
``categoryThicknessLimits``; ``fractionNewRidgeArea`` and
``fractionNewRidgeVolume`` are written only for the thinnest category's
ridges under the exponential redistribution and otherwise keep their words;
a ridging abort returns with the categories already ridged changed; the
transfer to the new ridges drops every tracer with ancestors other than the
brine fraction (the pond tracers start from zero on new ridges).

On an abort MPAS writes the aborting cell back, logs its message and returns
from the whole routine, so the later cells keep their words: :func:`ridging`
restores them and returns that cell and message. ``ridge_shift``'s two aborts
never get that far: v8.4.2 writes their 512-character stop label into a
256-character warning and dies with a gfortran runtime error, so
:func:`ridging` raises :class:`FortranRuntimeError` there.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal, MpasPortError

from . import column_itd_v842 as citd
from . import column_v842 as cv
from .velocity_v842 import COMPILE_OPTIONS, _grid

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:2498-2767",
    "src/core_seaice/column/ice_colpkg.F90:2981-3161",
    "src/core_seaice/column/ice_mechred.F90:80-1535",
) + citd.SOURCE_FILES[3:]
_BLOCK = 128
SEAICE_ERROR_COL_RIDGING = 12
#: ridge_prep's hin_max(ncat), which lands in categoryThicknessLimits
BIG = 1.0e8
#: the line ridge_ice logs per ridging iteration that leaves the area off 1
REPEAT_WARNING = "WARN  Repeat ridging, niter = {:11d}"

#: the stop labels by the kernel's abort code
STOP_LABELS = {
    1: "ridge_ice: Exceeded max number of ridging iterations",
    2: "Ridging error: aice0 < 0",
    3: "Ridging error: ardg > aicen",
    4: "ridge_ice: total area > 1",
    5: "ice: ITD cleanup error in colpkg_step_ridge",
}

#: ridge_shift's aborts write their stop label (column_ridging's abortMessage,
#: strKIND = 512 characters) list-directed into a char_len_long = 256 character
#: warning: v8.4.2 dies there with this gfortran runtime error, by abort code
RUNTIME_ERRORS = {
    2: {"file": "ice_mechred.F90", "line": 1257, "error": "End of record"},
    3: {"file": "ice_mechred.F90", "line": 1294, "error": "End of record"},
}

refuse_unported = citd.refuse_unported


class FortranRuntimeError(MpasPortError):
    """MPAS v8.4.2 itself terminates here (a gfortran runtime error); the
    port stops the same way instead of inventing an outcome."""

    def __init__(self, crash: dict, cell: int) -> None:
        self.crash = crash
        self.cell = cell
        super().__init__(f"At line {crash['line']} of file {crash['file']} (indexToCellID {cell}): "
                         f"Fortran runtime error: {crash['error']} (MPAS v8.4.2 terminates here)")


def module_source(n_ice_layers: int) -> str:
    body = files("woof.hex.mpas_seaice").joinpath("column_ridge_v842.cu").read_text(encoding="utf-8")
    return citd.module_source(n_ice_layers) + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], n_ice_layers: int):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(n_ice_layers), options, None,
                                   f"seaice_column_ridge_v842_l{n_ice_layers}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str, n_ice_layers: int):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options, n_ice_layers).get_function(name)


def ridging(pools: dict, configs: dict, state: "cv.ColumnState") -> tuple[int, int | None, str | None]:
    """``column_ridging``: returns ``(ierr, cell, message)``, ``(0, None,
    None)`` when no cell aborts, else the first aborting owned cell's
    ``indexToCellID`` and its logged message. ``state.warnings`` is the list
    of the cells' warnings this port grades, in log order: ridge_ice's
    ``Repeat ridging`` lines and zap_snow_temperature's out-of-bounds lines.
    Needs ``init_column_itd`` to have run (the thickness limits)."""
    check_config(configs)
    import cupy as cp

    d = state.dims
    refuse_unported(configs, d)
    n_cells, n_solve, ncat, nil = d["nCells"], d["nCellsSolve"], d["nCategories"], d["nIceLayers"]
    to = state.tracers
    nt = to.n_tracers
    c1, cc, cat1 = (n_cells + 1,), (n_cells + 1, ncat), (n_cells + 1, ncat, 1)
    need = cv._need
    t, ist, ta, rg = pools["tracers"], pools["icestate"], pools["tracers_aggregate"], pools["ridging"]
    of = pools["ocean_fluxes"]
    limits = need(pools["initial"], "categoryThicknessLimits", (ncat + 1,))
    step = pools["velocity_solver"].get("dynamicsTimeStep")
    if not isinstance(step, cp.ndarray) or step.dtype != np.float64 or step.size != 1:
        raise ValueError("dynamicsTimeStep must be a device float64 scalar")
    dt = float(cp.asnumpy(step).reshape(-1)[0])
    area, volume, snow = (need(t, k, cat1) for k in ("iceAreaCategory", "iceVolumeCategory",
                                                      "snowVolumeCategory"))
    cell_fields = [need(ist, "openWaterArea", c1), need(ta, "iceAreaCell", c1),
                   *(need(rg, k, c1) for k in ("areaLossRidge", "areaGainRidge", "iceVolumeRidged",
                                                "openingRateRidge")),
                   need(pools["ponds"], "pondFreshWaterFlux", c1), need(of, "oceanFreshWaterFlux", c1),
                   need(of, "oceanSaltFlux", c1), need(of, "oceanHeatFlux", c1)]
    category_fields = [need(rg, k, cc) for k in (
        "ridgeParticipationFunction", "ratioRidgeThicknessToIce", "fractionNewRidgeArea",
        "fractionNewRidgeVolume", "areaLossRidgeCategory", "areaGainRidgeCategory", "iceVolumeRidgedCategory",
        "raftingIceArea", "raftingIceVolume")]
    newly = cv._need_int(pools["biogeochemistry"], "newlyFormedIce", cc)
    tracer_fields = [cv._tracer_field(pools, name)[0] for name, _, _ in to.layout]
    written = [area, volume, snow, newly, *cell_fields, *category_fields, *tracer_fields]
    before = [a[:n_solve].copy() for a in written]

    trcrn = cp.zeros((max(n_solve, 1), ncat, nt), np.float64)
    for name, first, layers in to.layout:
        src, sub = cv._tracer_field(pools, name)
        cv._launch("seaice_gather_tracer_v842", n_solve * ncat * layers, trcrn, src, np.int32(first - 1),
                   np.int32(layers), np.int32(sub), np.int32(src.shape[2]), np.int32(nt), np.int32(ncat),
                   np.int32(n_solve))
    atrcrn = cp.zeros_like(trcrn)
    tidx = cp.asarray(np.array([to.index[k] - 1 for k in citd._TRACER_INDICES] + [to.nt_fbri - 1], np.int32))
    status, nwarn, nrepeat = (cp.zeros(max(n_solve, 1), np.int32) for _ in range(3))
    ints = cv._CICE_INT
    partic = ints["config_ridging_participation_function"][configs["config_ridging_participation_function"]]
    redist = ints["config_ridging_redistribution_function"][configs["config_ridging_redistribution_function"]]
    args = (
        status, nwarn, nrepeat, np.float64(dt), np.int32(configs["config_dynamics_subcycle_number"]),
        np.float64(configs["config_ridiging_efolding_scale"]), np.int32(partic), np.int32(redist),
        np.int32(ncat), np.int32(nt), np.int32(citd._flags(configs, state)), np.int32(n_solve), tidx,
        cp.asarray(to.parent_index), cp.asarray(np.ascontiguousarray(to.first_ancestor_mask)),
        cp.asarray(to.ancestor_number), cp.asarray(np.ascontiguousarray(to.ancestor_indices)), limits, trcrn,
        atrcrn, area, volume, snow, cell_fields[0], cell_fields[1], need(rg, "ridgeConvergence", c1),
        need(rg, "ridgeShear", c1), *cell_fields[2:], *category_fields, newly,
    )
    if n_solve:
        _function("seaice_column_ridging_v842", nil)(_grid(n_solve), (_BLOCK,), args)
    for name, first, layers in to.layout:
        dst, sub = cv._tracer_field(pools, name)
        citd._function("seaice_scatter_category_tracer_v842", nil)(
            _grid(max(n_solve * ncat * layers, 1)), (_BLOCK,),
            (dst, trcrn, np.int32(first - 1), np.int32(layers), np.int32(sub), np.int32(dst.shape[2]),
             np.int32(nt), np.int32(ncat), np.int32(n_solve)))
    codes = cp.asnumpy(status)[:n_solve]
    zaps = cp.asnumpy(nwarn)[:n_solve]
    repeats = cp.asnumpy(nrepeat)[:n_solve]
    bad = np.flatnonzero(codes)
    last = int(bad[0]) if bad.size else n_solve - 1
    # MPAS returns at the first abort: the later cells keep their words
    for array, saved in zip(written, before):
        array[last + 1:n_solve] = saved[last + 1:]
    if n_solve:
        limits[ncat] = BIG          # ridge_prep, in the first cell already
    state.warnings = [line for c in range(last + 1)
                      for line in ([REPEAT_WARNING.format(k) for k in range(1, int(repeats[c]) + 1)]
                                   + [citd.ZAP_WARNING] * int(zaps[c]))]
    if not bad.size:
        return 0, None, None
    cell = int(cp.asnumpy(pools["mesh"]["indexToCellID"][last:last + 1])[0])
    if int(codes[last]) in RUNTIME_ERRORS:
        raise FortranRuntimeError(RUNTIME_ERRORS[int(codes[last])], cell)
    return SEAICE_ERROR_COL_RIDGING, cell, "column_ridging: " + STOP_LABELS[int(codes[last])]


#: the names v8.4.2's column package maps to colpkg's integers (anything else
#: aborts ``seaice_column_init`` there); the ridging kernels and
#: ``colpkg_ice_strength``'s carry exactly these
NAMED_OPTIONS = ("config_ice_strength_formulation", "config_ridging_participation_function",
                 "config_ridging_redistribution_function")


def check_config(configs: dict) -> None:
    """Refuse, by name and before any pool is touched, a strength formulation
    or a ridging participation or redistribution function that is missing or
    not one v8.4.2 names."""
    for knob in NAMED_OPTIONS:
        names = cv._CICE_INT[knob]
        value = configs.get(knob)
        if value not in names:
            raise ConfigurationRefusal(
                knob=knob, value=value, reason=f"v8.4.2's column package names only {sorted(names)}",
                declaration=" or ".join(f"{knob} = '{name}'" for name in names))


def ice_strength(pools: dict, configs: dict, n_ice_layers: int) -> None:
    """``ice_strength`` under the column package's vertical thermodynamics,
    in place (``mpas_seaice_velocity_solver.F:1359-1379``): ``icePressure``
    of the owned cells zeroed, then, where ``solveStress`` is 1,
    ``colpkg_ice_strength``'s (``config_ice_strength_formulation``:
    ``Rothrock75``, the Registry default, from ``ridge_itd``'s participation
    and ridges with ``config_ridging_participation_function``,
    ``config_ridging_redistribution_function`` and
    ``config_ridiging_efolding_scale``; or ``Hibler79``). Its ``Cf`` is the
    column constants' 17, as v8.4.2 does not read
    ``config_ratio_ridging_work_to_PE``. RKIND 8 only."""
    check_config(configs)
    ints = cv._CICE_INT
    strength = configs["config_ice_strength_formulation"]
    solver, aggregate = pools["velocity_solver"], pools["tracers_aggregate"]
    pressure = solver["icePressure"]
    n_cells = pressure.shape[0]
    area, volume = pools["tracers"]["iceAreaCategory"], pools["tracers"]["iceVolumeCategory"]
    ncat = area.shape[1]
    n_solve = int(pools["mesh"]["nCellsSolve"])
    if not 0 <= n_solve <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    if ncat > citd.ITD_MAXCAT:
        raise ValueError(f"colpkg_ice_strength here carries at most {citd.ITD_MAXCAT} categories")
    for name, a, shape in (("icePressure", pressure, (n_cells,)),
                           ("iceAreaCell", aggregate["iceAreaCell"], (n_cells,)),
                           ("iceVolumeCell", aggregate["iceVolumeCell"], (n_cells,)),
                           ("openWaterArea", pools["icestate"]["openWaterArea"], (n_cells,))):
        cv._need({name: a}, name, shape)
    for name, a in (("iceAreaCategory", area), ("iceVolumeCategory", volume)):
        if a.size != n_cells * ncat:
            raise ValueError(f"{name} must hold nCategories values per cell")
        cv._need({name: a.reshape(n_cells, ncat)}, name, (n_cells, ncat))
    cv._need_int({"solveStress": solver["solveStress"]}, "solveStress", (n_cells,))
    partic = ints["config_ridging_participation_function"][configs["config_ridging_participation_function"]]
    redist = ints["config_ridging_redistribution_function"][configs["config_ridging_redistribution_function"]]
    if n_solve:
        _function("seaice_column_ice_strength_v842", n_ice_layers)(_grid(n_solve), (_BLOCK,), (
            pressure, solver["solveStress"], aggregate["iceAreaCell"], aggregate["iceVolumeCell"],
            pools["icestate"]["openWaterArea"], area, volume, np.int32(ncat),
            np.int32(ints["config_ice_strength_formulation"][strength]), np.int32(partic), np.int32(redist),
            np.float64(configs["config_ridiging_efolding_scale"]), np.int32(n_solve)))
