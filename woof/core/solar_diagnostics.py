"""WRF-Solar diagnostics (``solar_diagnostics = 1``), on the device.

WRF v4.7.1 phys/module_diag_solar.F (T. Juliano and P. Jimenez, NCAR/RAL),
called by module_diagnostics_driver.F:1181-1213 at the end of EVERY time
step.  The routine itself is woof/core/kernels/solar_diag.cu, graded word
for word against the compiled routine (tools/solar_diag_wrf471_oracle).

The 26 outputs are Registry/registry.solar_fields' 2-D ``h`` rows, named and
described as WRF names them (:data:`SOLAR_OUTPUTS`).  None is a restart
field (the Registry gives no ``r``): GHI_ACCUM, the one accumulator, sums
SWDNB*dt from the start of THIS run segment and a WRF restart begins it at
zero; woof does the same (the plane is not serialized).  It is never reset
by a history write.

Inputs, as WRF hands them over (see :func:`update`):

* the end-of-step moisture (qv/qc/qi/qs), geopotential (PH, PHB) and the
  microphysics' effective radii (woof keeps them in microns; WRF's
  RE_CLOUD/RE_ICE/RE_SNOW are metres -- :func:`radius_metres`);
* the physics density and dz8w of the step's own physics preparation (WRF
  computes them in phy_prep at the start of the step);
* the radiation driver's CLDFRA and its qc/qi after the cloud scheme
  (QC_TOT/QI_TOT) and COSZEN -- which WRF's radiation driver recomputes on
  EVERY step when solar_diagnostics = 1 (its Solar_step block,
  module_radiation_driver.F:1176-1597, and :3103-3135), not only on
  radiation steps; woof evaluates them each step for the diagnostics
  (:mod:`woof.core.solar_step`) but keeps the forecast's own CLDFRA and
  COSZEN on the radiation cadence, so switching the diagnostics on changes no
  forecast word (WRF's surface, PBL and cumulus drivers do see the per-step
  values: a documented difference);
* SWDNB/SWDNT, the last radiation call's boundary fluxes.
"""
from __future__ import annotations

import numpy as np

#: (scratch slot, WRF name, units, description) in solar_diag.cu's plane
#: order.  Registry/registry.solar_fields, verbatim.
SOLAR_OUTPUTS = (
    ("solar_clrnidx", "CLRNIDX", " ", "CLEARNESS INDEX"),
    ("solar_sza", "SZA", "deg", "SOLAR ZENITH ANGLE"),
    ("solar_ghi_accum", "GHI_ACCUM", "J m-2", "ACCUMULATED GHI"),
    ("solar_cldfrac2d", "CLDFRAC2D", "%", "2-D MAX CLOUD FRACTION"),
    ("solar_wvp", "WVP", "kg m-2", "WATER VAPOR PATH"),
    ("solar_lwp", "LWP", "kg m-2", "LIQUID CLOUD WATER PATH"),
    ("solar_iwp", "IWP", "kg m-2", "ICE CLOUD WATER PATH"),
    ("solar_swp", "SWP", "kg m-2", "SNOW CLOUD WATER PATH"),
    ("solar_wp_sum", "WP_SUM", "kg m-2", "SUM OF LWP+IWP+SWP"),
    ("solar_lwp_tot", "LWP_TOT", "kg m-2", "LIQUID CLOUD WATER PATH RESOLVED + UNRESOLVED"),
    ("solar_iwp_tot", "IWP_TOT", "kg m-2", "ICE CLOUD WATER PATH RESOLVED + UNRESOLVED"),
    ("solar_wp_tot_sum", "WP_TOT_SUM", "kg m-2", "SUM OF LWP+IWP+SWP RESOLVED + UNRESOLVED"),
    ("solar_re_qc", "RE_QC", "m", "MASS-WEIGHTED LIQUID CLOUD EFFECTIVE RADIUS"),
    ("solar_re_qi", "RE_QI", "m", "MASS-WEIGHTED ICE EFFECTIVE RADIUS"),
    ("solar_re_qs", "RE_QS", "m", "MASS-WEIGHTED SNOW EFFECTIVE RADIUS"),
    ("solar_re_qc_tot", "RE_QC_TOT", "m", "MASS-WEIGHTED LIQUID CLOUD EFFECTIVE RADIUS RESOLVED + UNRESOLVED"),
    ("solar_re_qi_tot", "RE_QI_TOT", "m", "MASS-WEIGHTED ICE EFFECTIVE RADIUS RESOLVED + UNRESOLVED"),
    ("solar_tau_qc", "TAU_QC", " ", "MASS-WEIGHTED LIQUID CLOUD OPTICAL THICKNESS"),
    ("solar_tau_qi", "TAU_QI", " ", "MASS-WEIGHTED ICE OPTICAL THICKNESS"),
    ("solar_tau_qs", "TAU_QS", " ", "MASS-WEIGHTED SNOW OPTICAL THICKNESS"),
    ("solar_tau_qc_tot", "TAU_QC_TOT", " ", "MASS-WEIGHTED LIQUID CLOUD OPTICAL THICKNESS RESOLVED + UNRESOLVED"),
    ("solar_tau_qi_tot", "TAU_QI_TOT", " ", "MASS-WEIGHTED ICE OPTICAL THICKNESS RESOLVED + UNRESOLVED"),
    ("solar_cbaseht", "CBASEHT", "m agl", "CLOUD BASE HEIGHT"),
    ("solar_ctopht", "CTOPHT", "m agl", "CLOUD TOP HEIGHT"),
    ("solar_cbaseht_tot", "CBASEHT_TOT", "m agl", "CLOUD BASE HEIGHT RESOLVED + UNRESOLVED"),
    ("solar_ctopht_tot", "CTOPHT_TOT", "m agl", "CLOUD TOP HEIGHT RESOLVED + UNRESOLVED"),
)
#: slot -> (NAME, units, description), the shape the history builders take.
HISTORY = {slot: (name, units, description)
           for slot, name, units, description in SOLAR_OUTPUTS}
NOUT = len(SOLAR_OUTPUTS)

FLAG_QV, FLAG_QC, FLAG_QI, FLAG_QS = 1, 2, 4, 8
FLAG_REQC, FLAG_REQI, FLAG_REQS = 16, 32, 64


def species_flags(*, qv, qc, qi, qs, has_reqc, has_reqi, has_reqs):
    """solar_diag.cu's ``flags`` word from presence and the has_req* trio."""
    return (FLAG_QV * bool(qv) | FLAG_QC * bool(qc) | FLAG_QI * bool(qi)
            | FLAG_QS * bool(qs) | FLAG_REQC * bool(has_reqc)
            | FLAG_REQI * bool(has_reqi) | FLAG_REQS * bool(has_reqs))


def radius_metres(microns, out=None):
    """WRF's RE_* (metres) from woof's micron radii: ``microns / 1e6``.

    woof's microphysics publishes ``fl(re_m * 1e6)``, the RRTMG wrapper's
    micron word; WRF's grid holds ``re_m`` itself.  The round trip is not
    the identity everywhere: over every float32 radius from 2.49e-6 to
    2e-3 m, dividing by 1e6 returns 3,597,476 of 81,524,879 words one ULP
    off (multiplying by 1e-6: 4,212,052), never more.  So RE_*/TAU_* can sit
    one ULP (and what that ULP moves) from WRF's; the routine's arithmetic
    itself is graded word for word (tests/test_solar_diag_wrf471_oracle.py).
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    microns = cp.ascontiguousarray(microns, dtype=cp.float32)
    if out is None:
        out = cp.empty_like(microns)
    n = microns.size
    if n:
        get_kernel("solar_diag", "solar_radius_metres")(
            ((n + 255) // 256,), (256,), (np.int64(n), microns, out))
    return out


def run_solar_diag(*, nz, ncol, dt, flags, rho, dz8w, ph, phb, cldfra, coszen,
                   swdnb, swdnt, qv, qc, qi, qs, qc_tot, qi_tot, re_cloud,
                   re_ice, re_snow, out, work):
    """Launch solar_diag.cu over ``ncol`` columns.

    Mass fields are ``(nz, ncol)`` float32 (column fastest), ``ph``/``phb``
    ``(nz+1, ncol)``, the surface fields ``(ncol,)``.  ``out`` is the
    ``(NOUT, ncol)`` output stack in :data:`SOLAR_OUTPUTS` order; its
    GHI_ACCUM plane is read and advanced, and an output WRF does not assign
    keeps its word.  ``work`` is ``(nz, ncol)`` float32 scratch.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    if ncol == 0:
        return out
    if nz < 2:
        raise ValueError("solar_diag integrates kts..kte-1 and needs nz >= 2")
    args = []
    for name, a, shape in (
            ("rho", rho, (nz, ncol)), ("dz8w", dz8w, (nz, ncol)),
            ("ph", ph, (nz + 1, ncol)), ("phb", phb, (nz + 1, ncol)),
            ("cldfra", cldfra, (nz, ncol)), ("coszen", coszen, (ncol,)),
            ("swdnb", swdnb, (ncol,)), ("swdnt", swdnt, (ncol,)),
            ("qv", qv, (nz, ncol)), ("qc", qc, (nz, ncol)),
            ("qi", qi, (nz, ncol)), ("qs", qs, (nz, ncol)),
            ("qc_tot", qc_tot, (nz, ncol)), ("qi_tot", qi_tot, (nz, ncol)),
            ("re_cloud", re_cloud, (nz, ncol)), ("re_ice", re_ice, (nz, ncol)),
            ("re_snow", re_snow, (nz, ncol))):
        if (a.dtype != cp.float32 or a.size != int(np.prod(shape))
                or not a.flags.c_contiguous):
            raise ValueError(f"solar_diag: {name} must be contiguous float32 {shape}")
        args.append(a)
    if out.dtype != cp.float32 or out.size != NOUT * ncol or not out.flags.c_contiguous:
        raise ValueError("solar_diag: out must be contiguous float32 (NOUT, ncol)")
    if work.dtype != cp.float32 or work.size < nz * ncol:
        raise ValueError("solar_diag: work must hold nz*ncol float32")
    block = 128
    get_kernel("solar_diag", "solar_diag_columns")(
        ((ncol + block - 1) // block,), (block,),
        (np.int32(ncol), np.int32(nz), np.float32(dt), np.int32(flags),
         *args, work, out))
    return out


# ---------------------------------------------------------------------------
# The WRF driver around the routine: state slots, the per-step Solar_step
# evaluation and the end-of-step call.
# ---------------------------------------------------------------------------

#: (NOUT, ny, nx): the 26 output planes, the history fields' storage.  Cross-
#: step (GHI_ACCUM sums, unassigned outputs keep their word) and NOT
#: checkpointed -- WRF's Registry gives them no restart I/O, so a resumed run
#: starts GHI_ACCUM at zero as a WRF restart does.
OUT_SLOT = "solar_out"
#: Written at the physics step's start (WRF's phy_prep and radiation-driver
#: Solar_step), read by the end-of-step call of the same step.
STEP_MASS_SLOTS = ("solar_step_rho", "solar_step_dz", "solar_step_cldfra",
                   "solar_step_qc_tot", "solar_step_qi_tot")
STEP_PLANE_SLOTS = ("solar_step_coszen",)
#: End-of-step work: the radii in metres, the rhodz column, an absent
#: species' zeros, PHB broadcast when the base state is a 1-D profile.
WORK_MASS_SLOTS = ("solar_step_re_cloud", "solar_step_re_ice",
                   "solar_step_re_snow", "solar_step_work", "solar_step_zero")
WORK_FULL_SLOTS = ("solar_step_phb",)


def enabled(cfg) -> bool:
    return int(getattr(cfg, "solar_diagnostics", 0) or 0) == 1


def scratch_shapes(cfg):
    """Every scratch slot the diagnostics allocate, for memory planning."""
    if not enabled(cfg):
        return {}
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    shapes = {OUT_SLOT: (NOUT, ny, nx)}
    shapes.update({slot: (nz, ny, nx) for slot in STEP_MASS_SLOTS + WORK_MASS_SLOTS})
    shapes.update({slot: (ny, nx) for slot in STEP_PLANE_SLOTS})
    shapes.update({slot: (nz + 1, ny, nx) for slot in WORK_FULL_SLOTS})
    return shapes


def initialize(state, cfg):
    """Allocate the slots, zero (WRF's grid arrays start at zero)."""
    if not enabled(cfg):
        return
    for slot, shape in scratch_shapes(cfg).items():
        state.scratch(shape, slot)[...] = 0.0


def history_fields(state) -> dict:
    """WRF name -> (ny, nx) plane, views of the output stack; empty when the
    run does not carry the diagnostics."""
    existing = getattr(state, "existing_scratch", None)
    stack = (existing(OUT_SLOT) if existing is not None
             else getattr(state, "_scratch", {}).get(OUT_SLOT))
    if stack is None:
        return {}
    return {name: stack[i] for i, (_slot, name, _u, _d) in enumerate(SOLAR_OUTPUTS)}


def _shortwave_adapter(radiation):
    sw = getattr(radiation, "shortwave_adapter", None)
    return radiation if sw is None else sw


def driver_clouds(state, cfg, atmosphere, fields, *, cldfra_out, qc_out, qi_out):
    """WRF's radiation-driver cloud block, evaluated now.

    module_radiation_driver.F's ICLOUD arms (:1300-1510) as its Solar_step
    runs them on every step under solar_diagnostics = 1: CLDFRA from
    cal_cldfra1 (icloud = 1, then MYNN's subgrid cloud, :1403-1429),
    cal_cldfra2 (icloud = 2) or cal_cldfra3 (icloud = 3, with its made-up
    condensate), on the driver's own unclamped moisture; the qc/qi it
    leaves are QC_TOT/QI_TOT (:3114-3134).  The same routines the RRTMG
    adapters radiate with (tests/test_solar_diagnostics_wiring.py holds the
    radiation-step values to theirs).  Writes ``(nz, ny, nx)`` outputs.
    """
    import cupy as cp
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
    from woof.core import rrtmgp as _rrtmgp
    from woof.core.mynn_radiation import (merge_mynn_bl_clouds,
                                          mynn_bl_cloud_active, wrf_itimestep)
    nz, ny, nx = atmosphere["pressure"].shape
    ncol = ny * nx
    zero = state.scratch((nz, ny, nx), "solar_step_zero")
    species = {name: (getattr(state, name, None)
                      if getattr(state, name, None) is not None else zero)
               for name in ("qc", "qi", "qs")}
    qc_out[...] = species["qc"]
    qi_out[...] = species["qi"]
    grid = lambda a: cp.ascontiguousarray(a).reshape(nz, ncol)
    qv = grid(atmosphere["qv"])
    t = grid(atmosphere["temperature"])
    p = grid(atmosphere["pressure"])
    qc, qi, qs = qc_out.reshape(nz, ncol), qi_out.reshape(nz, ncol), grid(species["qs"])
    mp_physics = int(getattr(cfg, "mp_physics", 0))
    icloud = int(getattr(cfg, "icloud", 1))
    f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)
    cf = cldfra_out.reshape(nz, ncol)
    if icloud == 1:
        cf[...] = _rrtmgp.cal_cldfra1(qv, qc, qi, qs, t, p, f_qc=True, f_qi=f_qi,
                                      f_qs=f_qs, mp_physics=mp_physics)
    elif icloud == 2:
        cf[...] = _rrtmgp.cal_cldfra2(cp.ascontiguousarray(qc), cp.ascontiguousarray(qi),
                                      f_qc=mp_physics != 0, f_qi=f_qi)
    elif icloud == 3:
        from woof.core.cldfra3 import cal_cldfra3, cldfra3_gridkm
        from woof.core.rrtm_lw import _stream_is_capturing
        cols = lambda a: cp.ascontiguousarray(a.T)
        c3, q3c, q3i = cal_cldfra3(
            cols(qv), cols(qc), cols(qi), cols(qs),
            cols(grid(atmosphere["dz"])), cols(p), cols(t),
            fields["xland"].reshape(-1), cldfra3_gridkm(cfg.dx, cfg.dy),
            check=not _stream_is_capturing(cp))
        cf[...] = c3.T
        qc[...] = q3c.T
        qi[...] = q3i.T
    else:
        cf[...] = 0.0
    if icloud == 1 and mynn_bl_cloud_active(getattr(cfg, "bl_pbl_physics", 0),
                                            getattr(cfg, "icloud_bl", 0)):
        merge_mynn_bl_clouds(
            qc, qi, cf, qc_bl=grid(fields["qc_bl"]), qi_bl=grid(fields["qi_bl"]),
            cldfra_bl=grid(fields["cldfra_bl"]),
            bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
            icloud_bl=getattr(cfg, "icloud_bl", 0),
            itimestep=wrf_itimestep(state.elapsed_seconds, cfg.dt),
            bl_mynn_version=getattr(cfg, "bl_mynn_version", "wrf_461"),
            temperature=t)


def capture_solar_step(physics, state, cfg, atmosphere):
    """The step's start: WRF's phy_prep density and dz8w, and the radiation
    driver's Solar_step cloud and COSZEN (every step, radiation or not)."""
    if not enabled(cfg):
        return
    import cupy as cp
    nz, ny, nx = atmosphere["pressure"].shape
    m = (nz, ny, nx)
    state.scratch(m, "solar_step_rho")[...] = atmosphere["rho"]
    state.scratch(m, "solar_step_dz")[...] = atmosphere["dz"]
    driver_clouds(state, cfg, atmosphere, physics.fields,
                  cldfra_out=state.scratch(m, "solar_step_cldfra"),
                  qc_out=state.scratch(m, "solar_step_qc_tot"),
                  qi_out=state.scratch(m, "solar_step_qi_tot"))
    adapter = _shortwave_adapter(physics.radiation_callable)
    state.scratch((ny, nx), "solar_step_coszen")[...] = cp.asarray(
        adapter.coszen_now(state, cfg), dtype=cp.float32).reshape(ny, nx)


def update(state, cfg):
    """The end of the step (module_diagnostics_driver.F:1181-1213)."""
    if not enabled(cfg) or getattr(state, "physics", None) is None:
        return
    import cupy as cp
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    physics = state.physics
    nz1, ny, nx = state.php.shape
    nz = nz1 - 1
    m = (nz, ny, nx)
    zero = state.scratch(m, "solar_step_zero")
    present = {name: getattr(state, name, None) is not None
               for name in ("qv", "qc", "qi", "qs")}
    has_req = legacy_scheme_has_req(int(cfg.mp_physics), getattr(cfg, "use_mp_re", 1))
    radii, have = {}, []
    for flag, src, slot in zip(has_req, ("effc", "effi", "effs"),
                               ("solar_step_re_cloud", "solar_step_re_ice",
                                "solar_step_re_snow")):
        value = getattr(state, src, None)
        if flag and value is not None:
            radii[src] = radius_metres(value, out=state.scratch(m, slot))
            have.append(True)
        else:
            radii[src] = zero
            have.append(False)
    phb = state.phb
    if phb.ndim == 1:
        full = state.scratch((nz + 1, ny, nx), "solar_step_phb")
        full[...] = phb[:, None, None]
        phb = full
    flags = species_flags(qv=present["qv"], qc=present["qc"], qi=present["qi"],
                          qs=present["qs"], has_reqc=have[0], has_reqi=have[1],
                          has_reqs=have[2])
    sp = lambda name: getattr(state, name) if present[name] else zero
    run_solar_diag(
        nz=nz, ncol=ny * nx, dt=cfg.dt, flags=flags,
        rho=state.scratch(m, "solar_step_rho"),
        dz8w=state.scratch(m, "solar_step_dz"), ph=state.php, phb=phb,
        cldfra=state.scratch(m, "solar_step_cldfra"),
        coszen=state.scratch((ny, nx), "solar_step_coszen"),
        swdnb=physics.fields["swdnb"], swdnt=physics.fields["swdnt"],
        qv=sp("qv"), qc=sp("qc"), qi=sp("qi"), qs=sp("qs"),
        qc_tot=state.scratch(m, "solar_step_qc_tot"),
        qi_tot=state.scratch(m, "solar_step_qi_tot"),
        re_cloud=radii["effc"], re_ice=radii["effi"], re_snow=radii["effs"],
        out=state.scratch((NOUT, ny, nx), OUT_SLOT),
        work=state.scratch(m, "solar_step_work"))
