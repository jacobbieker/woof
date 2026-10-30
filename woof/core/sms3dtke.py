"""WRF v4.7.1 ``km_opt = 5``: the SMS-3DTKE scale-adaptive 3-D TKE closure.

Launchers for the ``sms3dtke`` kernel unit (woof/core/kernels/sms3dtke.cu),
one per WRF routine, plus :func:`run_step`, the once-per-step sequence the
dycore calls.  The kernels take WRF's own operands (th_phy, rho, rdzw, rdz,
zx, zy, the p8w/t8w surface and top planes, u_phy/v_phy), which
:func:`prepare` forms from woof's state the way phy_prep and
compute_diff_metrics do.

What WRF does with km_opt = 5 (diff_opt = 2, bl_pbl_physics = 0 only;
module_first_rk_step_part2.F):

* ``nonlocal_flux`` diagnoses the boundary-layer height with the theta
  method (it OVERWRITES ``PBLH`` and ``KPBL``), prescribes the LES-derived
  nonlocal heat flux profile ``NLFLUX`` scaled by the partial function
  pthnl(dx/zi), and the Brown-Grant nonlocal momentum gammas ``GAMU``/
  ``GAMV`` scaled by pu(dx/zi);
* ``free_atmos_length`` (BouLac) and ``meso_length_scale`` (Nakanishi-
  Niino, blended into BouLac above the PBL) give the mesoscale length
  ``DLK``;
* ``tke_km`` blends a Smagorinsky horizontal K with the TKE one by
  pthl(dx/zi) and an LES vertical K with the mesoscale one
  (``XKMV_MESO`` = 0.4 sqrt(e) DLK) by pu(dx/zi);
* ``tke_rhs`` adds shear and buoyancy production (the buoyancy carries the
  nonlocal flux) and writes the dissipation length ``L_DISS`` -- the
  dissipation itself is NOT added here;
* ``vertical_diffusion_implicit`` replaces vertical_diffusion_2: implicit
  vertical mixing of u, v (with the gamma counter-gradient terms and the
  ust surface stress), w (with xkmh), theta (with NLFLUX and HFX), TKE
  (2*xkmv, and the dissipation beta = 1.5 sqrt(e)/L_DISS treated
  implicitly), and every moist, scalar, tracer and chem row (QFX on vapour).

There is no PBL scheme: WRF refuses km_opt = 5 with bl_pbl_physics != 0
(share/module_check_a_mundo.F:357-367); the closure IS the column's
vertical mixing at every grid spacing, and its partial functions move it
from LES (dx <= 100 m: pthl = pthnl = pu = 0) to mesoscale (pu -> 1).
"""
from __future__ import annotations

import numpy as np

from woof.core.kernels import get_kernel

_TPB = 64
F32 = np.float32


def _k(name: str):
    return get_kernel("sms3dtke", name)


def _cols(nx: int, ny: int):
    return ((nx + _TPB - 1) // _TPB, ny, 1), (_TPB, 1, 1)


def _points(nx: int, ny: int, nz: int):
    return ((nx + _TPB - 1) // _TPB, ny, nz), (_TPB, 1, 1)


def _dims(nz, ny, nx, bx, by):
    return (np.int32(nz), np.int32(ny), np.int32(nx), np.int32(bool(bx)),
            np.int32(bool(by)))


def workspace_shape(nz: int, ny: int, nx: int) -> tuple[int, ...]:
    """The tridiagonal workspace: five planes of nz levels per column."""
    return (5 * nz * (ny + 1) * (nx + 1),)


def prepare(*, u, v, php, phb, thp, thb, alt, qv, p, rdx, rdy, bx, by, out):
    """phy_prep and compute_diff_metrics' operands into ``out`` (a dict of
    preallocated arrays: th, t2, rho, u_phy, v_phy, rdzw, rdz (nz, ny, nx),
    zx (nz+1, ny, nx+1), zy (nz+1, ny+1, nx), p8w1, t8w1, p8wt, t8wt
    (ny, nx))."""
    nz, ny, nx = thp.shape
    has_qv = qv is not None
    qv_arg = qv if has_qv else alt
    grid, block = _points(nx, ny, nz)
    _k("sms_prep_mass")(grid, block, (
        u, v, php, phb, np.int32(phb.ndim == 3), thp, thb,
        np.int32(thb.ndim == 3), alt, qv_arg, np.int32(has_qv),
        out["th"], out["t2"], out["rho"], out["u_phy"], out["v_phy"],
        out["rdzw"], out["rdz"], np.int32(nz), np.int32(ny), np.int32(nx)))
    grid, block = _points(nx + 1, ny + 1, nz + 1)
    _k("sms_prep_slope")(grid, block, (
        php, phb, np.int32(phb.ndim == 3), F32(rdx), F32(rdy),
        out["zx"], out["zy"], *_dims(nz, ny, nx, bx, by)))
    grid, block = _cols(nx, ny)
    _k("sms_prep_p8w")(grid, block, (
        p, out["th"], php, phb, np.int32(phb.ndim == 3),
        out["p8w1"], out["t8w1"], out["p8wt"], out["t8wt"],
        np.int32(nz), np.int32(ny), np.int32(nx)))


def nonlocal_flux(*, rdzw, th, qv, u_phy, v_phy, rho, ust, br, wspd, msft,
                  hfx, dx, dy, isfflx, tke_heat_flux, nlflux, gamu, gamv,
                  hpbl, kpbl, bx, by):
    nz, ny, nx = th.shape
    grid, block = _cols(nx, ny)
    _k("sms_nonlocal_flux")(grid, block, (
        rdzw, th, qv if qv is not None else th, np.int32(qv is not None),
        u_phy, v_phy, rho, ust, br, wspd, msft, hfx, F32(dx), F32(dy),
        np.int32(isfflx), F32(tke_heat_flux), nlflux, gamu, gamv, hpbl,
        kpbl, *_dims(nz, ny, nx, bx, by)))


def free_atmos_length(*, rdzw, tke, th, elmin, bx, by):
    nz, ny, nx = th.shape
    grid, block = _cols(nx, ny)
    _k("sms_free_atmos_length")(grid, block, (
        rdzw, tke, th, elmin, *_dims(nz, ny, nx, bx, by)))


def meso_length_scale(*, rdzw, rdz, tke, th, p8w1, t8w1, p8wt, t8wt, hpbl,
                      elmin, rmol, rho, hfx, qv, isfflx, tke_heat_flux, dlk,
                      bx, by):
    nz, ny, nx = th.shape
    grid, block = _cols(nx, ny)
    _k("sms_meso_length_scale")(grid, block, (
        rdzw, rdz, tke, th, p8w1, t8w1, p8wt, t8wt, hpbl, elmin, rmol, rho,
        hfx, qv if qv is not None else th, np.int32(qv is not None),
        np.int32(isfflx), F32(tke_heat_flux), dlk,
        *_dims(nz, ny, nx, bx, by)))


def tke_km(*, tke, th, rdz, rdzw, p8w1, t8w1, p8wt, t8wt, d11, d22, d12,
           zx, zy, msft, hpbl, dlk, dx, dy, c_s, c_k, tke_seed,
           xkmh, xkmv, xkhh, xkhv, xkmv_meso, bx, by):
    nz, ny, nx = th.shape
    grid, block = _points(nx, ny, nz)
    _k("sms_tke_km")(grid, block, (
        tke, th, rdz, rdzw, p8w1, t8w1, p8wt, t8wt, d11, d22, d12, zx, zy,
        msft, hpbl, dlk, F32(dx), F32(dy), F32(c_s), F32(c_k), F32(tke_seed),
        xkmh, xkmv, xkhh, xkhv, xkmv_meso, *_dims(nz, ny, nx, bx, by)))


def tke_rhs(*, tend, bn2, xkhv, nlflux, th, tke, mut, c1h, c2h, rdzw, msft,
            hpbl, dlk, qv, rho, hfx, isfflx, tke_heat_flux, c_k, dt, dx, dy,
            l_diss, bx, by):
    nz, ny, nx = th.shape
    grid, block = _points(nx, ny, nz)
    _k("sms_tke_rhs")(grid, block, (
        tend, bn2, xkhv, nlflux, th, tke, mut, c1h, c2h, rdzw, msft, hpbl,
        dlk, qv if qv is not None else th, np.int32(qv is not None), rho,
        hfx, np.int32(isfflx), F32(tke_heat_flux), F32(c_k), F32(dt),
        F32(dx), F32(dy), l_diss, *_dims(nz, ny, nx, bx, by)))


def vdiff_u(*, ru, u, v, rho, xkmv, xkmv_meso, gamu, rdz, fnm, fnp, dnw,
            c1h, c2h, mut, ustm, dt, ws, bx, by):
    nz, ny, nx = rho.shape
    grid, block = _cols(nx + 1, ny)
    _k("sms_vdiff_u")(grid, block, (
        ru, u, v, rho, xkmv, xkmv_meso, gamu, rdz, fnm, fnp, dnw, c1h, c2h,
        mut, ustm, F32(dt), ws, *_dims(nz, ny, nx, bx, by)))


def vdiff_v(*, rv, u, v, rho, xkmv, xkmv_meso, gamv, rdz, fnm, fnp, dnw,
            c1h, c2h, mut, ustm, dt, ws, bx, by):
    nz, ny, nx = rho.shape
    grid, block = _cols(nx, ny + 1)
    _k("sms_vdiff_v")(grid, block, (
        rv, u, v, rho, xkmv, xkmv_meso, gamv, rdz, fnm, fnp, dnw, c1h, c2h,
        mut, ustm, F32(dt), ws, *_dims(nz, ny, nx, bx, by)))


def vdiff_w(*, rw, w, rho, xkmh, rdzw, dn, c1f, c2f, mut, dt, ws, bx, by):
    nz, ny, nx = rho.shape
    grid, block = _cols(nx, ny)
    _k("sms_vdiff_w")(grid, block, (
        rw, w, rho, xkmh, rdzw, dn, c1f, c2f, mut, F32(dt), ws,
        *_dims(nz, ny, nx, bx, by)))


def vdiff_th(*, rt, var, th, rho, xkhv, nlflux, rdz, fnm, fnp, dnw, c1h,
             c2h, mut, hfx, qfx, qv, use_theta_m, isfflx, tke_heat_flux, dt,
             ws, bx, by):
    nz, ny, nx = rho.shape
    grid, block = _cols(nx, ny)
    _k("sms_vdiff_th")(grid, block, (
        rt, var, th, rho, xkhv, nlflux, rdz, fnm, fnp, dnw, c1h, c2h, mut,
        hfx, qfx, qv if qv is not None else th, np.int32(qv is not None),
        np.int32(use_theta_m), np.int32(isfflx), F32(tke_heat_flux),
        F32(dt), ws, *_dims(nz, ny, nx, bx, by)))


def vdiff_tke(*, rtke, tke, rho, xkmv, l_diss, rdz, fnm, fnp, dnw, c1h, c2h,
              mut, dt, ws, bx, by):
    nz, ny, nx = rho.shape
    grid, block = _cols(nx, ny)
    _k("sms_vdiff_tke")(grid, block, (
        rtke, tke, rho, xkmv, l_diss, rdz, fnm, fnp, dnw, c1h, c2h, mut,
        F32(dt), ws, *_dims(nz, ny, nx, bx, by)))


def vdiff_scalar(*, tend, field, base, rho, xkhv, rdz, fnm, fnp, dnw, c1h,
                 c2h, mut, flux, dt, ws, bx, by):
    nz, ny, nx = rho.shape
    grid, block = _cols(nx, ny)
    _k("sms_vdiff_scalar")(grid, block, (
        tend, field, base if base is not None else fnm,
        np.int32(base is not None), rho, xkhv, rdz, fnm, fnp, dnw, c1h, c2h,
        mut, flux if flux is not None else mut, np.int32(flux is not None),
        F32(dt), ws, *_dims(nz, ny, nx, bx, by)))


__all__ = ["free_atmos_length", "meso_length_scale", "nonlocal_flux",
           "prepare", "tke_km", "tke_rhs", "vdiff_scalar", "vdiff_th",
           "vdiff_tke", "vdiff_u", "vdiff_v", "vdiff_w", "workspace_shape"]
