"""Positive-definite moist transport TL/AD (P3a item 13, DA request 3):
WRF's ``moist_adv_opt = 1`` final RK stage on woof's strict path, one
species at a time (woof/core/moist.py ``_advance_species_wrf``, the
``pd`` branch; WRF solve_em.F:1850, 1876 rk_update_scalar_pd and
rk_scalar_tend's advect_scalar_pd, module_em.F and module_advect_em.F):

- the source fold (``_fold_wrf_pd_sources``; module_em.F
  rk_update_scalar_pd): ``q0e = ((chm0*q0) + (dt*src))/chm0`` with the
  time-t column mass ``chm0 = c1h*(mub + mup0) + c2h`` and the species'
  held source (zero without one);
- advect_scalar_pd (woof/core/kernels/pd_advection.cu ``pd_fluxes`` and
  ``pd_renorm_apply``, the GPUWM_WRF_EXACT_C_ADVECTION statements): the
  upwind flux of ``q0e``, the high-order correction of the stage's ``q``
  with the carried acoustic mass ``muts``, each cell's renormalisation
  factor against ``ph_low`` (``mub2d``, ``mup0``) and the limited flux
  divergence, the specified ring cleared;
- the update: ``q = ((chm0*q0e) + (dt*(tend*msft)))/(c1h*(mub + mup) + c2h)``,
  the fused ``gpuwm_update_scalar`` (no source: it was folded).

The limiter is linearised as WRFPLUS's G_/A_ADVECT_SCALAR_PD do: each
branch (Courant clamps, ``|cr|``, the flux_out max/min, the limiter test,
``max(0, ratio)``, the donor of each correction flux) follows the base
state.  The derivative does not exist where a perturbation switches one;
:func:`branches` gives the base state's branch words, for finite
difference checks that stay on one branch.

Same conventions as :mod:`woof.grad.wrfplus.scalar`: ``pd_tl(state,
pert_state, traj, pert, geom, nl=...)``, ``pd_ad(adj_state, traj, adj,
geom)`` adding into ``adj``; ``adj['q']`` may be the same array as
``adj_state['q']`` (the step's), so the update's seed is consumed first.
``pa_tl``/``pa_ad`` are the advection alone (the oracle's routine).

Not covered: ``v_sca_adv_order = 5`` (woof's strict PD runs the HRRR
fork's vertical arm, pd_vertical_sl.cu, not WRF 4.7.1's), h order 2,
open (radiative) boundaries (the primal runs the unlimited stage there),
IEVA and the RQVFTEN export.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._mode import base_only, base_only_active, tl_options
from .acoustic import _check, _dtype, _real
from .scalar import _ring

KERNELS = Path(__file__).resolve().parent / "kernels"
BLOCK = (32, 4, 1)
H_ORDERS = (3, 4, 5, 6)
V_ORDERS = (2, 3, 4, 6)

PD_STATE = ("q", "tend")
PD_ACTIVE = ("q", "q0", "ru_m", "rv_m", "ww_m", "mup0", "mup", "muts", "held")
PD_PASSIVE = ("mub2d", "c1h", "c2h", "msft", "rdnw", "fnm", "fnp")

#: the advection alone: tend from the stage's q and the folded q0e
PA_STATE = ("tend",)
PA_ACTIVE = ("q", "q0e", "ru_m", "rv_m", "ww_m", "muts", "mup0")
PA_PASSIVE = ("mub2d", "c1h", "c2h", "msft", "rdnw", "fnm", "fnp")
FLUXES = ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")


@dataclass(frozen=True)
class PDGeometry:
    """One species' PD stage: the grid, the primal's ``dx``/``dy`` (the
    kernels take ``DTYPE(dx)`` and ``DTYPE(1.0/dx)`` as the launchers
    do), the update's ``dt``, the orders, a specified domain (both axes
    bounded, the ring of ``spec_zone`` cleared), map factors and whether
    the species has a held source."""
    nz: int
    ny: int
    nx: int
    dx: float
    dy: float
    dt: float
    vorder: int = 3
    horder: int = 5
    spec: int = 0
    spec_zone: int = 0
    has_msf: int = 0
    held: int = 0

    @property
    def open_x(self) -> int:
        return self.spec

    @property
    def open_y(self) -> int:
        return self.spec

    def validate(self) -> None:
        if self.horder not in H_ORDERS:
            raise ValueError(f"horder must be one of {H_ORDERS}, got {self.horder!r}")
        if self.vorder not in V_ORDERS:
            raise ValueError(f"vorder must be one of {V_ORDERS} (5 is the HRRR fork's "
                             f"vertical arm), got {self.vorder!r}")
        for name in ("nz", "ny", "nx"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("spec", "has_msf", "held"):
            if getattr(self, name) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1, got {getattr(self, name)!r}")
        if self.spec and min(self.nx, self.ny) < 7:
            raise ValueError(f"a specified PD domain needs nx, ny >= 7, got "
                             f"{self.ny}x{self.nx}")
        if self.spec and not 0 < 2 * self.spec_zone <= min(self.ny, self.nx):
            raise ValueError(f"spec_zone {self.spec_zone} does not fit {self.ny}x{self.nx}")
        if not self.spec and self.spec_zone:
            raise ValueError("spec_zone needs spec")
        for name in ("dx", "dy", "dt"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")

    def reals(self, dtype) -> dict:
        r = np.dtype(dtype).type
        return dict(dx=r(self.dx), dy=r(self.dy), dt=r(self.dt),
                    dx_inv=r(1.0 / self.dx), dy_inv=r(1.0 / self.dy))

    def ints(self) -> tuple:
        i = np.int32
        return (i(self.nz), i(self.ny), i(self.nx), i(self.has_msf), i(self.open_x),
                i(self.open_y))


def pd_shapes(geom: PDGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    m, c = (nz, ny, nx), (ny, nx)
    return {"q": m, "tend": m, "q0": m, "q0e": m, "held": m, "ru_m": (nz, ny, nx + 1),
            "rv_m": (nz, ny + 1, nx), "ww_m": (nz + 1, ny, nx), "mup0": c, "mup": c,
            "muts": c, "mub2d": c, "msft": c, "c1h": (nz,), "c2h": (nz,), "rdnw": (nz,),
            "fnm": (nz,), "fnp": (nz,),
            "fxl": (nz, ny, nx + 1), "fxc": (nz, ny, nx + 1), "fyl": (nz, ny + 1, nx),
            "fyc": (nz, ny + 1, nx), "fzl": (nz + 1, ny, nx), "fzc": (nz + 1, ny, nx)}


def pd_active(geom: PDGeometry) -> tuple:
    return PD_ACTIVE if geom.held else tuple(n for n in PD_ACTIVE if n != "held")


def pa_active(geom: PDGeometry) -> tuple:
    return PA_ACTIVE


@lru_cache(maxsize=None)
def _device_module(real: str, horder: int, device: int, build: tuple = ()):
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    source = (KERNELS / "pdadv.cu").read_text(encoding="utf-8")
    options = ("--fmad=false", f"-DWRFPLUS_REAL={real}", f"-DADV_H={horder}") + build
    module = cp.RawModule(code=source, options=options)
    record_module("woof.grad.wrfplus.pdadv", source=source, options=options, module=module)
    return module


def _kernel(name: str, dtype, horder: int, build=None):
    import cupy as cp
    return _device_module(_real(dtype), horder, cp.cuda.Device().id,
                          tl_options() if build is None else build).get_function(name)


def _grid(ni: int, nj: int, nk: int) -> tuple:
    return ((ni + BLOCK[0] - 1) // BLOCK[0], (nj + BLOCK[1] - 1) // BLOCK[1], nk)


def _prepare(geom: PDGeometry, named) -> np.dtype:
    geom.validate()
    _check(geom, named, pd_shapes(geom), min_nz=1)
    return _dtype([a for _, _, a in named])


def _empty_fluxes(geom, dtype) -> dict:
    import cupy as cp
    sh = pd_shapes(geom)
    return {n: cp.empty(sh[n], dtype=dtype) for n in FLUXES}


def _advect(tend, tendd, tr, pe, geom: PDGeometry, dtype, build=None):
    """The three TL launches: fluxes, cell factors, limited divergence.
    ``tr`` holds the base arrays (PA names), ``pe`` their tangents (the
    base arrays again in the base-only build, which never reads them).
    Returns the base fluxes and factors (the adjoint's recompute)."""
    import cupy as cp
    re = geom.reals(dtype)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    f, fd = _empty_fluxes(geom, dtype), _empty_fluxes(geom, dtype)
    k = lambda n: _kernel(n, dtype, geom.horder, build)
    msft = tr["msft"]
    k("pd_flux_tl")(
        _grid(nx + 1, ny + 1, nz + 1), BLOCK,
        (tr["q"], pe["q"], tr["q0e"], pe["q0e"], tr["ru_m"], pe["ru_m"], tr["rv_m"],
         pe["rv_m"], tr["ww_m"], pe["ww_m"], tr["muts"], pe["muts"], tr["c1h"], tr["c2h"],
         tr["rdnw"], tr["fnm"], tr["fnp"], msft, re["dx"], re["dy"], re["dt"])
        + tuple(a for n in FLUXES for a in (f[n], fd[n]))
        + geom.ints() + (np.int32(geom.vorder),))
    sc, scd = cp.empty_like(tr["q"]), cp.empty_like(tr["q"])
    k("pd_scale_tl")(
        _grid(nx, ny, nz), BLOCK,
        (tr["q0e"], pe["q0e"], tr["mup0"], pe["mup0"], tr["mub2d"], tr["c1h"], tr["c2h"],
         tr["rdnw"], msft)
        + tuple(a for n in FLUXES for a in (f[n], fd[n]))
        + (re["dx_inv"], re["dy_inv"], re["dt"], sc, scd) + geom.ints())
    k("pd_apply_tl")(
        _grid(nx, ny, nz), BLOCK,
        tuple(a for n in FLUXES for a in (f[n], fd[n]))
        + (sc, scd, tr["rdnw"], msft, re["dx_inv"], re["dy_inv"], tend, tendd)
        + geom.ints())
    return f, sc


def _advect_base(tr, geom, dtype):
    """advect_scalar_pd's base tendency, fluxes and factors on ``tr``."""
    import cupy as cp
    tend = cp.empty_like(tr["q"])
    with base_only():
        f, sc = _advect(tend, tend, tr, tr, geom, dtype, tl_options())
    return tend, f, sc


def _advect_ad(at, tr, f, sc, adj, geom: PDGeometry, dtype) -> None:
    """Adjoint of :func:`_advect` from the tendency's adjoint ``at``: add
    into ``adj`` (PA_ACTIVE names) on the base ``tr``, fluxes ``f`` and
    factors ``sc``."""
    import cupy as cp
    re = geom.reals(dtype)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    k = lambda n: _kernel(n, dtype, geom.horder, ())
    msft = tr["msft"]
    a = _empty_fluxes(geom, dtype)
    g = {n: cp.empty_like(f[n]) for n in ("fxc", "fyc", "fzc")}
    faces = _grid(nx + 1, ny + 1, nz + 1)
    cells = _grid(nx, ny, nz)
    k("pd_apply_ad")(
        faces, BLOCK,
        (at, sc, f["fxc"], f["fyc"], f["fzc"], tr["rdnw"], msft, re["dx_inv"],
         re["dy_inv"], a["fxl"], a["fxc"], g["fxc"], a["fyl"], a["fyc"], g["fyc"],
         a["fzl"], a["fzc"], g["fzc"]) + geom.ints())
    aph, afo = cp.empty_like(at), cp.empty_like(at)
    aq0, amu = cp.empty_like(at), cp.empty_like(at)
    k("pd_scale_ad")(
        cells, BLOCK,
        (g["fxc"], g["fyc"], g["fzc"], tr["q0e"], tr["mup0"], tr["mub2d"], tr["c1h"],
         tr["c2h"], tr["rdnw"], msft) + tuple(f[n] for n in FLUXES)
        + (re["dx_inv"], re["dy_inv"], re["dt"], aph, afo, aq0, amu) + geom.ints())
    base = (tr["q"], tr["q0e"], tr["ru_m"], tr["rv_m"], tr["ww_m"], tr["muts"],
            tr["c1h"], tr["c2h"], tr["rdnw"], tr["fnm"], tr["fnp"], msft)
    k("pd_flux_ad_face")(
        faces, BLOCK,
        base + (f["fxc"], f["fyc"], f["fzc"], aph, afo, re["dx"], re["dy"], re["dt"],
                re["dx_inv"], re["dy_inv"])
        + tuple(a[n] for n in FLUXES) + (adj["ru_m"], adj["rv_m"], adj["ww_m"])
        + geom.ints() + (np.int32(geom.vorder),))
    amut = cp.empty_like(at)
    k("pd_flux_ad_cell")(
        cells, BLOCK,
        base + tuple(a[n] for n in FLUXES) + (re["dx"], re["dy"], re["dt"], adj["q"], aq0,
                                              amut) + geom.ints() + (np.int32(geom.vorder),))
    adj["q0e"] += aq0
    adj["muts"] += amut.sum(axis=0)
    adj["mup0"] += amu.sum(axis=0)


def _ones_msf(traj, geom):
    """The launchers' msft: the state's with map factors, else ones."""
    import cupy as cp
    if geom.has_msf:
        return traj["msft"]
    return cp.ones_like(traj["msft"])


def pa_tl(state, pert_state, traj, pert, geom: PDGeometry, *, nl=True) -> None:
    """TL of advect_scalar_pd (``tend`` = the limited flux divergence,
    overwritten, as the primal's zero_first launch): ``state['tend']``
    when ``nl``, ``pert_state['tend']`` from the perturbations of
    :data:`PA_ACTIVE`."""
    import cupy as cp
    tl = not base_only_active()
    names = [("state", "tend", state["tend"])]
    names += [("traj", n, traj[n]) for n in PA_ACTIVE + PA_PASSIVE]
    if tl:
        names += [("pert", "tend", pert_state["tend"])]
        names += [("pert", n, pert[n]) for n in PA_ACTIVE]
    dtype = _prepare(geom, names)
    tr = {**traj, "msft": _ones_msf(traj, geom)}
    tend = state["tend"] if nl else cp.empty_like(traj["q"])
    _advect(tend, pert_state["tend"] if tl else tend, tr, pert if tl else tr, geom, dtype)


def pa_ad(adj_state, traj, adj, geom: PDGeometry) -> None:
    """Adjoint of :func:`pa_tl`: ``adj_state['tend']`` (zeroed: the
    primal overwrites it) into ``adj`` (:data:`PA_ACTIVE`)."""
    names = [("adj", "tend", adj_state["tend"])]
    names += [("traj", n, traj[n]) for n in PA_ACTIVE + PA_PASSIVE]
    names += [("adj", n, adj[n]) for n in PA_ACTIVE]
    dtype = _prepare(geom, names)
    tr = {**traj, "msft": _ones_msf(traj, geom)}
    _, f, sc = _advect_base(tr, geom, dtype)
    at = adj_state["tend"].copy()
    adj_state["tend"][...] = 0
    _advect_ad(at, tr, f, sc, adj, geom, dtype)


# ------------------------------------------------------------ the stage

def _masses(traj, dtype):
    c1h = traj["c1h"][:, None, None]
    c2h = traj["c2h"][:, None, None]
    mu0 = traj["mub2d"] + traj["mup0"]
    mu = traj["mub2d"] + traj["mup"]
    chm0 = c1h * mu0[None] + c2h
    chm = c1h * mu[None] + c2h
    return c1h, chm0, chm


def _fold(traj, chm0, dt, geom):
    """``_fold_wrf_pd_sources``: ((chm0*q0) + (dt*src))/chm0, src the
    held source or the primal's zeros."""
    import cupy as cp
    src = traj["held"] if geom.held else cp.zeros_like(traj["q0"])
    return ((chm0 * traj["q0"]) + (dt * src)) / chm0


def _update(q0e, tend, traj, chm0, chm, dt, geom):
    """gpuwm_update_scalar: t = tend*msft; v = chm0*q0e; t = dt*t;
    v = v + t; v = v/chm (the fused kernel's rounded operations)."""
    t = tend * traj["msft"][None] if geom.has_msf else tend
    v = chm0 * q0e
    t = dt * t
    v = v + t
    return v / chm


def _pa_view(d, q0e) -> dict:
    out = {n: d[n] for n in PA_ACTIVE + PA_PASSIVE if n in d and n != "q0e"}
    out["q0e"] = q0e
    return out


def _stage_base(traj, geom, dtype):
    """The stage's base quantities on ``traj`` (the adjoint's recompute)."""
    dt = dtype.type(geom.dt)
    c1h, chm0, chm = _masses(traj, dtype)
    q0e = _fold(traj, chm0, dt, geom)
    tr = {**_pa_view(traj, q0e), "msft": _ones_msf(traj, geom)}
    tend, f, sc = _advect_base(tr, geom, dtype)
    if geom.spec:
        _ring(tend, geom.spec_zone)
    q = _update(q0e, tend, traj, chm0, chm, dt, geom)
    return dict(dt=dt, c1h=c1h, chm0=chm0, chm=chm, q0e=q0e, tr=tr, f=f, sc=sc, q=q)


def pd_tl(state, pert_state, traj, pert, geom: PDGeometry, *, nl=True) -> None:
    """TL of the species' PD stage: ``state['q']`` and the scratch
    ``state['tend']`` (the ring-cleared PD tendency) from the trajectory,
    ``pert_state['q']`` (and ``pert_state['tend']``) from the
    perturbations of :func:`pd_active`."""
    import cupy as cp
    tl = not base_only_active()
    names = [("state", n, state[n]) for n in PD_STATE]
    names += [("traj", n, traj[n]) for n in pd_active(geom) + PD_PASSIVE
              if n not in PD_STATE]
    if tl:
        names += [("pert", n, pert_state[n]) for n in PD_STATE]
        names += [("pert", n, pert[n]) for n in pd_active(geom) if n not in PD_STATE]
    dtype = _prepare(geom, names)
    dt = dtype.type(geom.dt)
    c1h, chm0, chm = _masses(traj, dtype)
    q0e = _fold(traj, chm0, dt, geom)
    tr = {**_pa_view(traj, q0e), "msft": _ones_msf(traj, geom)}
    tend = state["tend"] if nl else cp.empty_like(traj["q"])
    if tl:
        dchm0 = c1h * pert["mup0"][None]
        num = (dchm0 * traj["q0"] + chm0 * pert["q0"])
        if geom.held:
            num = num + dt * pert["held"]
        dq0e = (num - q0e * dchm0) / chm0
        pe = _pa_view(pert, dq0e)
        _advect(tend, pert_state["tend"], tr, pe, geom, dtype)
        if geom.spec:
            _ring(pert_state["tend"], geom.spec_zone)
    else:
        _advect(tend, tend, tr, tr, geom, dtype)
    if geom.spec:
        _ring(tend, geom.spec_zone)
    q = _update(q0e, tend, traj, chm0, chm, dt, geom)
    if tl:
        dt_t = pert_state["tend"] * traj["msft"][None] if geom.has_msf else pert_state["tend"]
        dv = dchm0 * q0e + chm0 * dq0e + dt * dt_t
        pert_state["q"][...] = (dv - q * (c1h * pert["mup"][None])) / chm
    if nl:
        state["q"][...] = q


def pd_ad(adj_state, traj, adj, geom: PDGeometry) -> None:
    """Adjoint of :func:`pd_tl`: ``adj_state['q']`` into ``adj``
    (:func:`pd_active`); ``adj_state['tend']`` is zeroed (overwritten by
    the primal).  ``traj`` is the base before the update."""
    import cupy as cp
    names = [("adj", n, adj_state[n]) for n in PD_STATE]
    names += [("traj", n, traj[n]) for n in pd_active(geom) + PD_PASSIVE]
    names += [("adj", n, adj[n]) for n in pd_active(geom) if n not in PD_STATE]
    dtype = _prepare(geom, names)
    b = _stage_base(traj, geom, dtype)
    dt, c1h, chm0, chm, q0e = b["dt"], b["c1h"], b["chm0"], b["chm"], b["q0e"]
    r = adj_state["q"] / chm
    adj_state["q"][...] = 0
    adj_state["tend"][...] = 0
    aq0e = chm0 * r
    amup0 = (c1h * q0e * r).sum(axis=0)
    adj["mup"] -= (c1h * b["q"] * r).sum(axis=0)
    at = dt * r
    if geom.has_msf:
        at *= traj["msft"][None]
    if geom.spec:
        _ring(at, geom.spec_zone)
    pa = {"q": adj["q"], "q0e": aq0e, "ru_m": adj["ru_m"], "rv_m": adj["rv_m"],
          "ww_m": adj["ww_m"], "muts": adj["muts"], "mup0": cp.zeros_like(amup0)}
    _advect_ad(at, b["tr"], b["f"], b["sc"], pa, geom, dtype)
    amup0 += pa["mup0"]
    # the fold: q0e = ((chm0*q0) + (dt*src))/chm0
    w = aq0e / chm0
    adj["q0"] += chm0 * w
    if geom.held:
        adj["held"] += dt * w
    amup0 += (c1h * (traj["q0"] - q0e) * w).sum(axis=0)
    adj["mup0"] += amup0


def branches(traj, geom: PDGeometry, *, stage=True) -> dict:
    """The base state's limiter branches (int8 words): the signs of the
    correction fluxes (the donor choice and flux_out's max/min) and of
    the mass fluxes (upwinding, dissipation), and each cell's state
    (0 unlimited, 1 limited with a positive ratio, 2 clipped to zero).
    ``stage``: ``traj`` holds the stage's names (folded here), else the
    advection's (``q0e`` given).  Two states whose words agree take the
    same branches; compare them to keep a finite difference on one."""
    import cupy as cp
    dtype = np.dtype(traj["q"].dtype)
    if stage:
        tr = _stage_base(traj, geom, dtype)["tr"]
    else:
        tr = {**traj, "msft": _ones_msf(traj, geom)}
    _, f, sc = _advect_base(tr, geom, dtype)
    out = {n: cp.sign(f[n]).astype(np.int8) for n in ("fxc", "fyc", "fzc")}
    for n in ("ru_m", "rv_m", "ww_m"):
        out[n] = cp.sign(tr[n]).astype(np.int8)
    state = cp.zeros(sc.shape, dtype=np.int8)
    state[sc != 1] = 1
    state[sc == 0] = 2
    out["cell"] = state
    return out
