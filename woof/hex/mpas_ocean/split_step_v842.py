"""MPAS-Ocean v8.4.2: one split-explicit step (``ocn_time_integrator_split``) on the GPU.

:func:`split_step` is ``ocn_time_integrator_split``
(``mpas_ocn_time_integration_split.F``) for one block, in the Fortran's order:

1. the preparation: the baroclinic velocity, the new time level started from
   the old;
2. ``config_n_ts_iter`` iterations, each: the momentum tendency
   (:func:`~woof.hex.mpas_ocean.forcing_v842.tend_vel`, at the old level
   first, then the new); stage 1's baroclinic iterations on the linear
   Coriolis term (``ocn_fuperp`` and the thickness-weighted barotropic
   forcing); stage 2, the barotropic subcycle
   (:mod:`~woof.hex.mpas_ocean.btr_subcycle_v842`); stage 3: the vertical
   transport (:mod:`~woof.hex.mpas_ocean.vert_transport_v842`), the thickness
   and tracer tendencies (:func:`~woof.hex.mpas_ocean.forcing_v842.tend_thick`,
   :mod:`~woof.hex.mpas_ocean.tend_tracer_v842`) and the update: half-way
   with ``ocn_diagnostic_solve`` before the last iteration, the full step (with
   the budgets per unit thickness and the salinity floor) at it;
3. the diagnostics and implicit vertical mixing
   (:mod:`~woof.hex.mpas_ocean.diagnostics_v842`,
   :mod:`~woof.hex.mpas_ocean.vmix_v842`), the prescribed velocity or
   thickness, the diagnostics again, the cell-centred velocity and SSH
   gradient (``mpas_reconstruct``), the coupler's surface fields and their
   running means (``ocn_time_average_coupled_accumulate``).

The pools are dicts of device arrays named as in MPAS (``mesh``, ``state``,
``tracers``, ``diagnostics``, ``forcing``, ``tend``, ``tracersTend``,
``tracersSurfaceFlux``, ``verticalMesh``), with MPAS's missing-neighbour
element; a field's time level 2 is ``NAME@2``; dimensions are host integers.
The step's own loops (``split_step_v842.cu``) and every piece are held word
for word to v8.4.2, the step cut verbatim and everything it calls compiled
verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_ocean_oracle/run_split_step.F90``,
``tests/test_mpas_ocean_split_step_parity.py``).

Refused by name, beyond what the pieces refuse: the high-frequency thickness
filter, GM (``config_use_standardGM``), the coupled land-ice effective
density, a time integrator other than 'split_explicit' or
'unsplit_explicit', nested halo extents (the momentum and thickness
tendencies run over the block), and the sphere's reconstruction at RKIND 8
without the glibc 2.39 opt-in (``WOOF_GLIBC239_LIBM64=1``).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_ice import rkind

from . import (btr_subcycle_v842, diagnostics_v842, forcing_v842, tend_tracer_v842, vert_transport_v842,
               vmix_v842)
from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = (
    "src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F:82-1926",
    "src/core_ocean/shared/mpas_ocn_diagnostics.F:1062-1128",
    "src/operators/mpas_vector_reconstruction.F:205-339,489-588",
    "src/core_ocean/shared/mpas_ocn_time_average_coupled.F:202-412",
)
MODULE_OPTIONS = COMPILE_OPTIONS
GRAVITY = 9.80616
T0_KELVIN = 273.16
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("split_step_v842.cu").read_text(encoding="utf-8")
    if precision == "r8":
        return "#define MPAS_RKIND8 1\n" + rkind.r8_trig_prelude() + body
    import woof.core.kernels as kernels
    trig = (Path(kernels.__file__).parent / "glibc_trig_flt32.cuh").read_text(encoding="utf-8")
    return trig + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...], trig64: bool):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"ocn_split_step_v842_{precision}{'_trig64' if trig64 else ''}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
        _module_on(cp.cuda.Device().id, precision, options, precision == "r8" and rkind.trig64()).get_function(
            name)(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict, precision: str, mesh: dict) -> None:
    """Refuse, by name, what this step does not carry."""
    integrator = configs["config_time_integrator"]
    if integrator not in ("split_explicit", "unsplit_explicit"):
        _refuse("config_time_integrator", integrator,
                "ocn_time_integrator_split leaves its split flag unset for any other integrator",
                "config_time_integrator = 'split_explicit' | 'unsplit_explicit'")
    if configs.get("config_use_freq_filtered_thickness", False):
        _refuse("config_use_freq_filtered_thickness", True, "the high-frequency thickness filter is not ported",
                "config_use_freq_filtered_thickness = .false.")
    if configs.get("config_use_standardGM", False):
        _refuse("config_use_standardGM", True, "GM (mpas_ocn_gm.F) is not ported",
                "config_use_standardGM = .false.")
    if configs.get("config_land_ice_flux_mode", "off") == "coupled":
        _refuse("config_land_ice_flux_mode", "coupled", "the coupled land-ice effective density is not ported",
                "config_land_ice_flux_mode = 'off' | 'pressure_only'")
    for name, count in (("nCellsArray", mesh["nCells"]), ("nEdgesArray", mesh["nEdges"])):
        if any(int(x) != int(count) for x in mesh[name]):
            _refuse(name, [int(x) for x in mesh[name]],
                    "the momentum and thickness tendencies here run over the whole block",
                    f"every {name} entry the block's count")
    if configs.get("mesh:on_a_sphere", False) and precision == "r8" and not rkind.trig64():
        _refuse("on_a_sphere", True, rkind.trig_refusal("mpas_reconstruct calls glibc's double cos and sin"),
                "on_a_sphere = .false. at RKIND 8, or WOOF_GLIBC239_LIBM64=1")


def _cut(pool: dict, sizes: dict) -> dict:
    """The pool with every cell, edge and vertex array cut to the block's count
    (the momentum and thickness tendencies size their loops by the arrays)."""
    out = {}
    for name, value in pool.items():
        on_device = value is not None and hasattr(value, "shape") and not isinstance(value, np.ndarray)
        n = sizes.get(int(value.shape[0])) if on_device and value.ndim else None
        out[name] = value[:n] if n is not None else value
    return out


def split_step(*, dt: float, n_btr_subcycles: int, pools: dict, configs: dict) -> None:
    """``ocn_time_integrator_split`` for one block, in place on the pools."""
    import cupy as cp
    m, st, tr = pools["mesh"], pools["state"], pools["tracers"]
    d, f, td = pools["diagnostics"], pools["forcing"], pools["tend"]
    tt, tsf, vm = pools["tracersTend"], pools["tracersSurfaceFlux"], pools["verticalMesh"]
    precision = _precision_of(st["layerThickness"])
    real = PRECISIONS[precision]
    check_config(configs, precision, m)
    i32 = np.int32
    nc, ne, nv, nl = int(m["nCells"]), int(m["nEdges"]), int(m["nVertices"]), int(m["nVertLevels"])
    me, me2 = int(m["edgesOnCell"].shape[1]), int(m["edgesOnEdge"].shape[1])
    nt = int(tr["activeTracers"].shape[2])
    rdt = real(dt)
    gravity = real(GRAVITY)
    integrator = configs["config_time_integrator"]
    split = 1 if integrator == "split_explicit" else 0

    def lv(pool: dict, name: str, level: int):
        return pool[name] if level == 1 else pool[f"{name}@2"]

    def state_at(level: int) -> dict:
        return {name: lv(st, name, level) for name in ("layerThickness", "normalVelocity", "ssh")}

    def tracers_at(level: int) -> dict:
        return {"activeTracers": lv(tr, "activeTracers", level), "index_temperature": tr["index_temperature"],
                "index_salinity": tr["index_salinity"]}

    sizes = {nc + 1: nc, ne + 1: ne, nv + 1: nv}
    mesh_cut = _cut(m, sizes)
    tend_configs = {k: configs[k] for k in forcing_v842.DEFAULT_CONFIGS if k in configs}

    # the preparation
    _launch(precision, "ocn_ss_prep_edges_v842", ne, (
        st["normalBaroclinicVelocity"], st["normalVelocity@2"], st["normalBaroclinicVelocity@2"],
        st["normalVelocity"], st["normalBarotropicVelocity"], i32(ne), i32(nl)))
    _launch(precision, "ocn_ss_prep_cells_v842", nc, (
        st["ssh@2"], st["layerThickness@2"], d["vertAleTransportTop"], tr["activeTracers@2"], st["ssh"],
        st["layerThickness"], tr["activeTracers"], m["maxLevelCell"], i32(nc), i32(nl), i32(nt)))

    n_ts = int(configs["config_n_ts_iter"])
    n_bcl = [int(configs["config_n_bcl_iter_mid"])] * n_ts
    n_bcl[0] = int(configs["config_n_bcl_iter_beg"])
    n_bcl[n_ts - 1] = int(configs["config_n_bcl_iter_end"])
    u_temp = cp.zeros((ne + 1) * nl, dtype=real)
    for step in range(1, n_ts + 1):
        if (configs.get("config_disable_thick_all_tend", False) and configs.get("config_disable_vel_all_tend", False)
                and configs.get("config_disable_tr_all_tend", False)):
            break
        level = min(step, 2)
        # stage 1: the momentum tendency and the baroclinic iterations
        forcing_v842.tend_vel(
            mesh=mesh_cut, state={"normalVelocity": lv(st, "normalVelocity", level)[:ne],
                                  "activeTracers": lv(tr, "activeTracers", level)[:nc]},
            diagnostics=_cut(d, sizes), forcing=_cut(f, sizes), tend=td["normalVelocity"][:ne],
            configs=tend_configs, index_temperature=int(tr["index_temperature"]),
            index_salinity=int(tr["index_salinity"]))
        if not (configs.get("config_disable_vel_all_tend", False) or configs.get("config_disable_vel_hmix", False)):
            # ocn_vel_hmix_tend zeroes the whole viscosity array, the missing edge's row too
            d["viscosity"][ne:] = 0
        for _ in range(n_bcl[step - 1]):
            _launch(precision, "ocn_ss_fuperp_v842", int(m["nEdgesSolve"]), (
                st["normalVelocity@2"], st["normalBaroclinicVelocity@2"], m["weightsOnEdge"], m["fEdge"],
                m["edgesOnEdge"], m["nEdgesOnEdge"], m["maxLevelEdgeTop"], i32(m["nEdgesSolve"]), i32(nl), i32(me2)))
            n1 = int(m["nEdgesArray"][0])
            _launch(precision, "ocn_ss_bcl_v842", n1, (
                st["normalBaroclinicVelocity@2"], d["barotropicForcing"], u_temp, st["normalBaroclinicVelocity"],
                td["normalVelocity"], st["normalVelocity@2"], st["ssh@2"], d["layerThicknessEdge"], m["dcEdge"],
                m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(n1), i32(nl), i32(split), rdt, gravity))
        # stage 2: the barotropic velocity
        btr_state = {name: [st[name], st[f"{name}@2"]] for name in (
            "normalBarotropicVelocity", "sshSubcycle", "normalBarotropicVelocitySubcycle", "normalVelocity",
            "normalBaroclinicVelocity")}
        btr_state["ssh"] = [st["ssh"]]
        btr_subcycle_v842.barotropic_stage(dt=dt, n_btr_subcycles=n_btr_subcycles, mesh=m, state=btr_state,
                                           diagnostics=d, tend=td, configs=configs)
        # stage 3: the vertical transport, the thickness and tracer tendencies, the update
        active_only = step < n_ts
        vert_transport_v842.vert_transport_velocity_top(
            mesh=m, resting_thickness=vm["restingThickness"], old_layer_thickness=st["layerThickness"],
            layer_thickness_edge=d["layerThicknessEdge"], normal_velocity=d["normalTransportVelocity"],
            old_ssh=st["ssh"], dt=dt, vert_ale_transport_top=d["vertAleTransportTop"], configs=configs)
        forcing_v842.tend_thick(mesh=mesh_cut, diagnostics=_cut(d, sizes), forcing=_cut(f, sizes),
                                tend=td["layerThickness"][:nc], configs=tend_configs)
        tend_tracer_v842.tend_tracer(dt=dt, mesh=m, state=state_at(2), tracers=tracers_at(2), tracers_tend=tt,
                                     tracers_surface_flux=tsf, diagnostics=d, forcing=f, configs=configs)
        del active_only  # one tracer group: the active tracers either way
        h_cur, h_new = st["layerThickness"], st["layerThickness@2"]
        tr_cur, tr_new = tr["activeTracers"], tr["activeTracers@2"]
        if step < n_ts:
            _launch(precision, "ocn_ss_mid_cells_v842", nc, (
                h_new, tr_new, h_cur, td["layerThickness"], tr_cur, tt["activeTracersTend"], m["maxLevelCell"],
                i32(nc), i32(nl), i32(nt), i32(tr["activeGRP_start"]), i32(tr["activeGRP_end"]), rdt))
            _launch(precision, "ocn_ss_mid_edges_v842", ne, (
                st["normalVelocity@2"], st["normalBarotropicVelocity@2"], st["normalBaroclinicVelocity@2"],
                m["edgeMask"], i32(ne), i32(nl)))
            diagnostics_v842.diagnostic_solve(dt=dt, mesh=m, state=state_at(2), tracers=tracers_at(2),
                                              diagnostics=d, forcing=f, configs=configs)
        else:
            _launch(precision, "ocn_ss_final_thickness_v842", nc, (
                h_new, h_cur, td["layerThickness"], m["maxLevelCell"], i32(nc), i32(nl), rdt))
            if configs.get("config_compute_active_tracer_budgets", False):
                for name, n in (("activeTracerHorizontalAdvectionTendency", nt),
                                ("activeTracerVerticalAdvectionTendency", nt),
                                ("activeTracerSurfaceFluxTendency", nt), ("temperatureShortWaveTendency", 1),
                                ("activeTracerNonLocalTendency", nt)):
                    _launch(precision, "ocn_ss_budget_v842", nc, (
                        d[name], h_new, m["maxLevelCell"], i32(nc), i32(nl), i32(n)))
            if configs.get("config_use_activeTracers", True):
                _launch(precision, "ocn_ss_final_tracers_v842", nc, (
                    tr_new, tr_cur, tt["activeTracersTend"], h_cur, h_new, m["maxLevelCell"], i32(nc), i32(nl),
                    i32(nt), i32(tr["index_salinity"]), rdt))
            _launch(precision, "ocn_ss_final_velocity_v842", ne, (
                st["normalVelocity@2"], st["normalBarotropicVelocity@2"], st["normalBaroclinicVelocity@2"],
                st["normalBaroclinicVelocity"], m["maxLevelEdgeTop"], i32(ne), i32(nl)))

    diagnostics_v842.diagnostic_solve(dt=dt, mesh=m, state=state_at(2), tracers=tracers_at(2), diagnostics=d,
                                      forcing=f, configs=configs)
    vmix_v842.vmix_implicit(dt=dt, mesh=m, state=state_at(2), tracers=tracers_at(2), diagnostics=d,
                            configs=configs)
    if configs.get("config_prescribe_velocity", False):
        st["normalVelocity@2"][:ne] = st["normalVelocity"][:ne]
    if configs.get("config_prescribe_thickness", False):
        st["layerThickness@2"][:nc] = st["layerThickness"][:nc]
    diagnostics_v842.diagnostic_solve(dt=dt, mesh=m, state=state_at(2), tracers=tracers_at(2), diagnostics=d,
                                      forcing=f, configs=configs)

    sphere = i32(bool(configs.get("mesh:on_a_sphere", False)))
    for u, names, levels in ((st["normalVelocity@2"], ("velocityX", "velocityY", "velocityZ", "velocityZonal",
                                                        "velocityMeridional"), nl),
                             (d["gradSSH"], ("gradSSHX", "gradSSHY", "gradSSHZ", "gradSSHZonal",
                                             "gradSSHMeridional"), 1)):
        _launch(precision, "ocn_ss_reconstruct_v842", nc, (
            *(d[n] for n in names), u, m["coeffs_reconstruct"], m["nEdgesOnCell"], m["edgesOnCell"], m["latCell"],
            m["lonCell"], i32(nc), i32(levels), i32(me), sphere))
    _launch(precision, "ocn_ss_surface_v842", nc, (
        d["surfaceVelocity"], d["SSHGradient"], d["velocityZonal"], d["velocityMeridional"], d["gradSSHZonal"],
        d["gradSSHMeridional"], i32(nc), i32(nl), i32(d["index_surfaceVelocityZonal"]),
        i32(d["index_surfaceVelocityMeridional"]), i32(d["index_SSHGradientZonal"]),
        i32(d["index_SSHGradientMeridional"])))
    n_cells_forcing = int(f["nCells"])
    _launch(precision, "ocn_ss_time_average_v842", n_cells_forcing, (
        f["avgTracersSurfaceValue"], f["avgSSHGradient"], f["avgSurfaceVelocity"], d["tracersSurfaceValue"],
        d["gradSSHZonal"], d["gradSSHMeridional"], d["surfaceVelocity"], i32(n_cells_forcing),
        i32(f["avgTracersSurfaceValue"].shape[1]), i32(f["index_avgTemperatureSurfaceValue"]),
        i32(f["index_avgSSHGradientZonal"]), i32(f["index_avgSSHGradientMeridional"]),
        i32(f["nAccumulatedCoupled"]), real(T0_KELVIN)))
