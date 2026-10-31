"""Deng shallow cumulus and cumulus-radiation feedback on the MPAS-A v8.4.2
physics path (WOOF-only services).

MPAS v8.4.2 has no shallow-convection scheme and its radiation reads no
convective cloud (the Grell-Freitas ``qc_cu``/``qi_cu`` it writes are
diagnostics).  These are WRF v4.7.1's ``shcu_physics = 5`` and
``cu_rad_feedback`` carried to MPAS as two services in the physics-suite
profile (``mpasphys_v842.services``), both default ``off``, selected in
``&hex_mpasphys``:

``config_shcu_deng_backend = 'hex-cuda'``
    The physics driver calls ``shcu_deng`` on every step right after the
    convection block (WRF's shallowcu_driver follows cumulus_driver), and
    ``shcu_deng_tend`` once per step inside ``physics_get_tend``.

    * ``shcu_deng`` hands every owned column to the WRF-graded Deng column
      body (woof/core/kernels/shcu_deng.cu, held to compiled WRF v4.7.1 by
      tests/test_shcu_deng_wrf471_oracle.py) through its MPAS entry point
      ``deng_shcu_mpas_column``.  The words are MPAS's own, from the
      prepared arrays MPAS_to_physics fills every step: ``u_p v_p t_p qv_p
      qc_p qr_p`` (WRF U V T QV QC QR), ``pres_p`` (P), ``pi_p`` (PI),
      ``rho_p`` (RHO, moist as WRF's), ``dz_p`` (DZ8W), ``z_p``/``w_p`` at
      the w levels (Z_AT_W, W); MYNN's ``qke`` (halved by the column body as
      WRF's driver does for bl_pbl_physics = 5), ``kzh`` (EXCH_H),
      ``el_pbl``; ``hpbl`` (PBLH); ``zgrid(1)`` (HT, as MPAS's convection
      takes it); ``lonCell/degrad`` (XLONG, the MPAS radiation's
      convention); ``rthratenlw/sw``; ``mavail`` (updated in place, as WRF
      does); the step's minutes ``time_seconds/60`` and the manager's
      ``gmt``.  Two quantities have no MPAS word and are defined here:

      - DX per cell: MPAS's own column-scheme grid length,
        ``config_len_disp / meshDensity**0.25``
        (mpas_atmphys_driver_convection.F:718);
      - DSIGMA (WRF's DNW, used by the cloud-water settling) per cell: the
        WRF eta increment ``dnw(k) = (pd(k+1) - pd(k)) / (pd(1) - pd(top))``
        of the column's dry hydrostatic pressure at the w levels,
        ``pres2_hydd_p`` -- WRF's terrain-following eta is exactly that
        normalized dry pressure (hybrid_opt = 0).

      The scheme's state (W0AVG, TKEAVG, the cloud-area/liquid leapfrog
      pairs, CLDFRA_SH/CA_RAD/CW_RAD, PBLMAX ... AINCKFSA) lives in the
      service, cold-started as deng_shcu_init does, and travels in the
      physics restart state.  W0AVG is Deng's own: MPAS's Kain-Fritsch keeps
      its own w0avg on its own cadence, so the KF/Deng sharing of WRF's grid
      field is not reproduced.
    * ``shcu_deng_tend`` adds the step's Deng tendencies to the sources
      physics_get_tend assembled, as one more coupled term in
      mpas_atmphys_todynamics.F's own arithmetic: with ``m`` the dry mass
      ``rho_zz``, ``th = RTHSHTEN*m``, ``qv = RQVSHTEN*m``,
      ``coeff = 1 + R_v/R_d*qv_state``,
      ``tend_rtheta += coeff*th + R_v/R_d*theta_m*qv/coeff``, and
      ``tend_scalars(qv/qc/qr) += RQ*SHTEN*m``, on owned cells.  (WRF's
      update_phy_ten adds RTHSHTEN/RQVSHTEN/RQCSHTEN/RQRSHTEN beside the
      cumulus tendencies; Deng's RUSHTEN/RVSHTEN are zero.)
    * the radiation's cloud: on radiation steps the hex-cuda cloudiness slot
      applies module_radiation_driver.F:1566-1575 to the cloud it forms,
      ``cldfrac = MAX(CLDFRA_SH, cldfrac)``, ``qcrad = CW_RAD + qcrad``
      (woof.core.cumulus_radiation.merge_deng_cloud), so both MPAS RRTMG
      schemes, which read ``cldfrac`` and ``qcrad_p``, see it.

``config_cu_rad_feedback_backend = 'hex-cuda'``
    WRF's ICLOUD_CU = 1 (module_radiation_driver.F:1264-1281) for MPAS's
    Grell-Freitas: before the cloud fraction, the cloudiness slot's copies
    become ``qc = qc + qc_cu``, ``qi = qi + qi_cu``
    (woof.core.cumulus_radiation.add_convective_condensate) from the
    ``diag_physics`` ``qc_cu``/``qi_cu`` cu_grell_freitas writes.  (WRF
    averages GD_CLOUD in convtrans_prep under cu_diag = 1; MPAS's
    Grell-Freitas stores its cloud directly, and that is the word used.)

FARMS (``config_farms_sw_backend``) takes the step's condensate through the
same additions, in WRF's order, before its cloud optics.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields, replace

import numpy as np

F = np.float32
SERVICE = "shcu_deng"
FEEDBACK_SERVICE = "cu_rad_feedback"
OPERATION = "shcu_deng"
COUPLE = "shcu_deng_tend"
#: The MPAS cumulus schemes whose cloud cu_rad_feedback hands to radiation.
FEEDBACK_SCHEMES = ("cu_grell_freitas",)
#: Deng reads MYNN's TKE/EXCH_H/EL_PBL; MPAS v8.4.2 has no MYJ.
DENG_PBL = "bl_mynn"
#: WRF bl_pbl_physics of MPAS's bl_mynn, for the TKE convention.
MYNN = 5
TENDENCIES = ("rthshten", "rqvshten", "rqcshten", "rqrshten")


def _cp():
    import cupy
    return cupy


def service_on(config, name: str) -> bool:
    from .driver_v842 import Backend
    return getattr(config, "service_backends", {}).get(name, Backend.OFF) != Backend.OFF


def _launch(name, count, args):
    from woof.core.shcu_deng import _deng_function
    if count:
        _deng_function(name)(((count + 127) // 128,), (128,), args)


def mpas_surface(cp, *, mesh_density, lon, len_disp, degrad):
    """MPAS's column-scheme grid length ``len_disp / meshDensity**0.25``
    and ``XLONG = lonCell/degrad`` per cell, each (nc,)."""
    density = cp.ascontiguousarray(cp.asarray(mesh_density, dtype=cp.float32))
    lon = cp.ascontiguousarray(cp.asarray(lon, dtype=cp.float32))
    nc = int(density.size)
    dx = cp.empty(nc, dtype=cp.float32)
    xlong = cp.empty(nc, dtype=cp.float32)
    _launch("deng_mpas_surface", nc, (np.int32(nc), F(len_disp), F(degrad),
                                       density, lon, dx, xlong))
    return dx, xlong


def mpas_dnw(cp, pres2_hydd, *, nl: int, nc: int):
    """WRF's DNW per cell from the dry hydrostatic w-level pressure,
    (nc, nl): ``(pd(k+1) - pd(k)) / (pd(1) - pd(nl+1))``."""
    pd = cp.ascontiguousarray(cp.asarray(pres2_hydd, dtype=cp.float32)[:nl + 1, :nc])
    dnw = cp.empty((nc, nl), dtype=cp.float32)
    _launch("deng_mpas_dnw", nc, (np.int32(nl), np.int32(nc), np.int32(nc), pd, dnw))
    return dnw


def cold_state(nl: int, nc: int):
    from woof.core.shcu_deng import DengState
    return DengState.cold(nl, 1, nc)


def deng_mpas(cp, *, nl, nc, state, u, v, t, qv, qc, qr, pres, pii, rho, dz,
              z_w, w_w, qke, kzh, el_pbl, rthratenlw, rthratensw, hpbl, ht,
              lon, degrad, mavail, mesh_density, len_disp, pres2_hydd, dt,
              xtime_min, gmt, perturb=None):
    """One Deng call on MPAS [level, cell] arrays: the first ``nc`` cells,
    levels 1..nl (w-level arrays 1..nl+1).  ``state`` (nl, 1, nc) advances
    in place and ``mavail`` (owned cells first) is updated in place.
    Returns launch_deng's outputs, each (nl, 1, nc)."""
    from woof.core.shcu_deng import launch_deng

    def prof(a, n=nl):
        return cp.ascontiguousarray(
            cp.asarray(a, dtype=cp.float32)[:n, :nc]).reshape(n, 1, nc)

    def surf(a):
        return cp.ascontiguousarray(
            cp.asarray(a, dtype=cp.float32)[:nc]).reshape(1, nc)

    moist = surf(mavail)
    dx, xlong = mpas_surface(cp, mesh_density=cp.asarray(mesh_density)[:nc],
                             lon=cp.asarray(lon)[:nc], len_disp=len_disp,
                             degrad=degrad)
    dx, xlong = dx.reshape(1, nc), xlong.reshape(1, nc)
    dnw = mpas_dnw(cp, pres2_hydd, nl=nl, nc=nc).reshape(1, nc, nl)
    qv_in, qc_in, w_in = prof(qv), prof(qc), prof(w_w, nl + 1)
    if perturb is not None:
        # WOOF-only WRF-Solar EPS (wrfsolar_eps_v842, config_pert_deng): WRF's Add_ on the copies.
        perturb(qv=qv_in, qc=qc_in, w=w_in)
    out = launch_deng(
        u=prof(u), v=prof(v), t=prof(t), qv=qv_in, qc=qc_in, qr=prof(qr),
        p=prof(pres), pii=prof(pii), rho=prof(rho), dz8w=prof(dz),
        tke=prof(qke), exch_h=prof(kzh), el_pbl=prof(el_pbl),
        rthratenlw=prof(rthratenlw), rthratensw=prof(rthratensw),
        z_at_w=prof(z_w, nl + 1), w=w_in, pblh=surf(hpbl),
        ht=surf(ht), xlong=xlong, dnw=dnw, mavail=moist, state=state,
        dt=float(dt), dx=dx, xtime=float(xtime_min), gmt=float(gmt),
        adaptive=False, bl_pbl_physics=MYNN)
    mavail[:nc] = moist.reshape(nc)
    return out


def _eps_deng(session):
    if getattr(session, "wrfsolar_eps", None) is None:
        return None
    from .wrfsolar_eps_v842 import deng_inputs
    return lambda **copies: deng_inputs(session, **copies)


def deng_step(provider, call, service):
    """The ``shcu_deng`` glue operation on the session's device pool views."""
    cp = provider.cp
    dims = provider.session.layout.dimensions
    nl, nc = int(dims["nVertLevels"]), int(dims["nCellsSolve"])
    prep, diag = provider["prepared"], provider["diag_physics"]
    tend, mesh = provider["tend_physics"], provider["mesh"]
    service._ensure_state()
    gmt = provider.native.get_table("mpas_atmphys_manager.gmt")[0][()]
    out = deng_mpas(
        cp, nl=nl, nc=nc, state=service.state, u=prep["u_p"], v=prep["v_p"],
        t=prep["t_p"], qv=prep["qv_p"], qc=prep["qc_p"], qr=prep["qr_p"],
        pres=prep["pres_p"], pii=prep["pi_p"], rho=prep["rho_p"], dz=prep["dz_p"],
        z_w=prep["z_p"], w_w=prep["w_p"], qke=diag["qke"], kzh=diag["kzh"],
        el_pbl=diag["el_pbl"], rthratenlw=tend["rthratenlw"],
        rthratensw=tend["rthratensw"], hpbl=diag["hpbl"], ht=mesh["zgrid"][0],
        lon=mesh["lonCell"], degrad=provider["constants"]["degrad"],
        mavail=diag["mavail"], mesh_density=mesh["meshDensity"],
        len_disp=provider["configs"]["config_len_disp"],
        pres2_hydd=prep["pres2_hydd_p"], dt=call.dt,
        xtime_min=F(call.time_seconds) / F(60.0), gmt=gmt,
        perturb=_eps_deng(provider.session))
    service.tendencies = {n: cp.ascontiguousarray(out[n].reshape(nl, nc))
                          for n in TENDENCIES}
    service.rainsh = out.get("rainsh")
    return out


def couple_deng(cp, *, rho_theta, scalars, mass, theta_m, qv, tendencies,
                index_qv, index_qc, index_qr, nc):
    """Add the Deng tendencies to physics_get_tend's sources on the owned
    cells ``[:, :nc]``, in mpas_atmphys_todynamics.F's arithmetic (module
    docstring; kernel deng_mpas_couple).  ``rho_theta`` (nl, ncells) and
    ``scalars`` (ns, nl, ncells), C-contiguous device arrays, are updated in
    place; ``mass``/``theta_m``/``qv`` are read on ``[:nl, :nc]``."""
    from woof.hex.cuda_physics_v841 import RV_OVER_RD_F32
    nl, dstride = (int(n) for n in rho_theta.shape)
    if not (rho_theta.flags.c_contiguous and scalars.flags.c_contiguous
            and tuple(scalars.shape[1:]) == (nl, dstride)):
        raise ValueError("Deng coupling needs C-contiguous sources of one layout")

    def state(a):
        return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32)[:nl, :nc])
    t = {n: cp.ascontiguousarray(tendencies[n], dtype=cp.float32) for n in TENDENCIES}
    _launch("deng_mpas_couple", nl * nc, (
        np.int32(nl), np.int32(nc), np.int32(nc), np.int32(dstride),
        np.int32(nl * dstride), np.int32(index_qv), np.int32(index_qc),
        np.int32(index_qr), F(RV_OVER_RD_F32), state(mass), state(theta_m),
        state(qv), t["rthshten"], t["rqvshten"], t["rqcshten"], t["rqrshten"],
        rho_theta, scalars))


def convective_cloud(session):
    """The convective cloud the cloudiness slot and FARMS merge, or None
    entries: ``deng`` (cldfra_sh, cw_rad) as (nl, nc) and ``feedback``."""
    service = getattr(session, "shcu_deng_service", None)
    deng = None
    if service is not None:
        service._ensure_state()
        nl, _, nc = service.state.cldfra_sh.shape
        deng = (service.state.cldfra_sh.reshape(nl, nc),
                service.state.cw_rad.reshape(nl, nc))
    return {"deng": deng, "feedback": service_on(session.config, FEEDBACK_SERVICE)}


def merged_condensate(cp, *, qc, qi, qc_cu, qi_cu, cw_rad, f_qc, f_qi):
    """WRF's radiation-driver additions to the condensate, in order:
    ICLOUD_CU = 1 (:1264-1281), then Deng's QC = CW_RAD + QC (:1566-1575).
    Any of qc_cu/qi_cu (feedback off) or cw_rad (Deng off) may be None."""
    from woof.core import cumulus_radiation as curad
    if qc_cu is not None:
        qc, qi = curad.add_convective_condensate(qc, qi, qc_cu, qi_cu,
                                                 f_qc=f_qc, f_qi=f_qi)
    if cw_rad is not None:
        qc = cw_rad + qc
    return qc, qi


class DengService:
    """Binds ``shcu_deng``/``shcu_deng_tend`` to a physics session."""

    def __init__(self, session):
        self.session = session
        self.state = None
        self.tendencies = None
        self.rainsh = None

    def step(self, call):
        provider = self.session.device_pools
        provider.begin(call, {})
        try:
            return deng_step(provider, call, self)
        finally:
            provider.commit()

    def couple(self, call):
        from woof.hex.cuda_physics_v841 import CudaMpasStageSourcesV842  # noqa: F401
        cp = _cp()
        sources = call.pools["sources"]
        if self.tendencies is None:
            return sources
        session = self.session
        nc = int(session.layout.dimensions["nCellsSolve"])
        names = tuple(next(row["components"] for row in session.layout.fields
                           if row["pool"] == "state" and row["name"] == "scalars"))
        host = not isinstance(sources.rho_theta, cp.ndarray)
        rho_theta = cp.ascontiguousarray(cp.array(sources.rho_theta, dtype=cp.float32))
        scalars = cp.ascontiguousarray(cp.array(sources.scalars, dtype=cp.float32))
        state_scalars = cp.asarray(session.field("state", "scalars"), dtype=cp.float32)
        couple_deng(cp, rho_theta=rho_theta, scalars=scalars,
                    mass=session.field("state", "rho_zz", level=2),
                    theta_m=session.field("state", "theta_m"),
                    qv=state_scalars[names.index("qv")], tendencies=self.tendencies,
                    index_qv=names.index("qv"), index_qc=names.index("qc"),
                    index_qr=names.index("qr"), nc=nc)
        if host:
            rho_theta, scalars = cp.asnumpy(rho_theta), cp.asnumpy(scalars)
        return replace(sources, rho_theta=rho_theta, scalars=scalars)

    def callbacks(self):
        return {OPERATION: self.step, COUPLE: self.couple}

    def _ensure_state(self):
        if self.state is None:
            dims = self.session.layout.dimensions
            self.state = cold_state(int(dims["nVertLevels"]), int(dims["nCellsSolve"]))
        return self.state

    def restart_state(self):
        """The Deng fields WRF keeps in its grid (and restart file) between
        steps; the tendencies are recomputed every step before they are read."""
        cp = _cp()
        state = self._ensure_state()
        return {f.name: cp.asnumpy(getattr(state, f.name)) for f in dc_fields(state)}

    def restore_restart_state(self, payload):
        from woof.core.shcu_deng import DengState
        cp = _cp()
        current = self._ensure_state()
        names = {f.name for f in dc_fields(current)}
        if set(payload) != names:
            raise ValueError("Deng service restart state differs")
        for name in names:
            mine = getattr(current, name)
            if tuple(payload[name].shape) != tuple(mine.shape) or payload[name].dtype != mine.dtype:
                raise ValueError(f"Deng service restart state layout differs at {name}")
        self.state = DengState(**{n: cp.asarray(payload[n]) for n in names})
        self.tendencies = None


__all__ = ["COUPLE", "DengService", "FEEDBACK_SCHEMES", "FEEDBACK_SERVICE",
           "OPERATION", "SERVICE", "convective_cloud", "couple_deng", "deng_mpas",
           "deng_step", "merged_condensate", "mpas_dnw", "mpas_surface", "service_on"]
