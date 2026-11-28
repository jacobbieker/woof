"""Moist scalar transport TL/AD (P3a 8j, DA item 4): WRF's sumflux and
one species' rk_scalar_tend + rk_update_scalar, non-positive-definite
(``moist_adv_opt = 0``), on woof's strict path:

- sumflux (woof/core/dycore.py ``_step_body``; the ``zero_sumflux``,
  ``accumulate_sumflux`` and ``finish_sumflux`` kernels of
  woof/core/kernels/acoustic.cu, WRF module_small_step_em.F sumflux):
  ``ru_m, rv_m, ww_m`` start at zero, add ``u_pp, v_pp, ww_pp`` after each
  acoustic substep, and end as the substep mean plus the stage's coupled
  fluxes ``ru, rv, ww``;
- the update (woof/core/moist.py ``_advance_species_wrf``, non-PD branch;
  WRF rk_scalar_tend + rk_update_scalar, module_em.F): ``tend`` is
  flux_div_scalar of the stage's ``q`` with the sumflux fluxes, the
  specified ring cleared, times ``msft``, plus the species' held
  source (boundary tendency or mixing row), and
  ``q = ((c1h*mu0 + c2h)*q0 + dt*tend) / (c1h*mu + c2h)`` with the
  time-t mass ``mu0 = mub + mup0`` and the post-acoustic ``mu = mub + mup``.

The arithmetic is the primal's, one rounded operation at a time (the
fused ``gpuwm_update_scalar`` kernel is compiled ``-fmad=false``), so the
float build gives the primal's words.  Same conventions as
:mod:`woof.grad.wrfplus.bigstep`: ``*_tl(state, pert_state, traj, pert,
geom, nl=...)`` and ``*_ad(adj_state, traj, adj, geom)``, the adjoint
adding into ``adj``.

The positive-definite final stage (``moist_adv_opt = 1``) is
:mod:`woof.grad.wrfplus.pdadv`.  Not covered: physics and mixing sources
(``moist_tend`` holds only the boundary rows here), IEVA, and the RQVFTEN
export.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._mode import base_only_active
from .acoustic import _check, _dtype
from .advect import AdvGeometry, adv_ad, adv_tl

#: sumflux's accumulators, what each substep adds and the stage fluxes
#: the finish adds to the mean.
SUMFLUX = ("ru_m", "rv_m", "ww_m")
SUMFLUX_SRC = ("u_pp", "v_pp", "ww_pp")
SUMFLUX_REF = ("ru", "rv", "ww")

SU_STATE = ("q", "tend")
SU_ACTIVE = ("q", "q0", "ru_m", "rv_m", "ww_m", "mup0", "mup", "held")
SU_PASSIVE = ("mub2d", "c1h", "c2h", "msft", "rdnw", "fnm", "fnp")


def _tangent() -> bool:
    """Whether the perturbation is formed (not in model.step_base)."""
    return not base_only_active()


@dataclass(frozen=True)
class SumfluxGeometry:
    nz: int
    ny: int
    nx: int
    nsub: int

    def validate(self) -> None:
        for name in ("nz", "ny", "nx", "nsub"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


def sumflux_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    u, v, w = (nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx)
    return dict(zip(SUMFLUX + SUMFLUX_SRC + SUMFLUX_REF, (u, v, w) * 3))


def _sf_check(geom, named) -> np.dtype:
    """Shapes, contiguity, distinct buffers and one real type."""
    geom.validate()
    _check(geom, named, sumflux_shapes(geom), min_nz=1)
    return _dtype([a for _, _, a in named])


def _sf_named(geom, state=None, pert_state=None, traj=None, pert=None, srcs=()):
    out = []
    for label, d, names in (("state", state, SUMFLUX), ("pert", pert_state, SUMFLUX),
                            ("traj", traj, srcs), ("pert", pert, srcs)):
        if d is not None:
            out += [(label, n, d[n]) for n in names]
    return out


def sumflux_zero_tl(state, pert_state, geom: SumfluxGeometry, *, nl=True) -> None:
    """``zero_sumflux``."""
    _sf_check(geom, _sf_named(geom, state, pert_state if _tangent() else None))
    for n in SUMFLUX:
        if nl:
            state[n][...] = 0
        if _tangent():
            pert_state[n][...] = 0


def sumflux_zero_ad(adj_state, geom: SumfluxGeometry) -> None:
    _sf_check(geom, _sf_named(geom, adj_state))
    for n in SUMFLUX:
        adj_state[n][...] = 0


def sumflux_accumulate_tl(state, pert_state, traj, pert, geom: SumfluxGeometry,
                          *, nl=True) -> None:
    """``accumulate_sumflux``: ``ru_m += u_pp`` (and v, ww)."""
    _sf_check(geom, _sf_named(geom, state, *((pert_state,) if _tangent() else (None,)),
                              traj, pert if _tangent() else None, srcs=SUMFLUX_SRC))
    for m, s in zip(SUMFLUX, SUMFLUX_SRC):
        if nl:
            state[m] += traj[s]
        if _tangent():
            pert_state[m] += pert[s]


def sumflux_accumulate_ad(adj_state, traj, adj, geom: SumfluxGeometry) -> None:
    _sf_check(geom, _sf_named(geom, adj_state, traj=adj, srcs=SUMFLUX_SRC))
    for m, s in zip(SUMFLUX, SUMFLUX_SRC):
        adj[s] += adj_state[m]


def sumflux_finish_tl(state, pert_state, traj, pert, geom: SumfluxGeometry,
                      *, nl=True) -> None:
    """``finish_sumflux``: ``ru_m = ru_m / nsub + ru`` (and v, ww), the
    division and the add each rounded."""
    dt = _sf_check(geom, _sf_named(geom, state, *((pert_state,) if _tangent() else (None,)),
                                   traj, pert if _tangent() else None, srcs=SUMFLUX_REF))
    nsub = dt.type(geom.nsub)
    for m, r in zip(SUMFLUX, SUMFLUX_REF):
        if nl:
            state[m][...] = state[m] / nsub + traj[r]
        if _tangent():
            pert_state[m][...] = pert_state[m] / nsub + pert[r]


def sumflux_finish_ad(adj_state, traj, adj, geom: SumfluxGeometry) -> None:
    nsub = _sf_check(geom, _sf_named(geom, adj_state, traj=adj, srcs=SUMFLUX_REF)).type(geom.nsub)
    for m, r in zip(SUMFLUX, SUMFLUX_REF):
        adj[r] += adj_state[m]
        adj_state[m][...] = adj_state[m] / nsub


@dataclass(frozen=True)
class ScalarGeometry:
    """One species' stage update: the grid, ``rdx``/``rdy``, the update's
    ``dt`` (the primal's ``DTYPE(nsub*dtau)``), the advection's orders
    and lateral options, the map factor switch, the ring width on a
    specified domain, and whether the species has a held source
    (``held``, on any domain: qv's boundary tendency with a table, or the
    species' diff_opt 1 mixing row, which moist._wrf_scalar_source
    holds)."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    dt: float
    vorder: int = 3
    horder: int = 5
    open_x: int = 0
    open_y: int = 0
    spec: int = 0
    spec_zone: int = 0
    has_msf: int = 0
    held: int = 0

    def adv(self) -> AdvGeometry:
        return AdvGeometry("scalar", self.nz, self.ny, self.nx, self.rdx, self.rdy,
                           open_x=self.open_x, open_y=self.open_y, spec=self.spec,
                           vorder=self.vorder, horder=self.horder)

    def validate(self) -> None:
        self.adv().validate()
        for name in ("open_x", "open_y", "spec", "has_msf", "held"):
            if getattr(self, name) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1, got {getattr(self, name)!r}")
        if self.spec and not 0 < 2 * self.spec_zone <= min(self.ny, self.nx):
            raise ValueError(f"spec_zone {self.spec_zone} does not fit "
                             f"{self.ny}x{self.nx}")
        if not self.spec and self.spec_zone:
            raise ValueError("spec_zone needs spec")
        if self.spec and not (self.open_x and self.open_y):
            # the ring is cleared on all four sides, as the primal does on
            # a specified domain (both axes bounded)
            raise ValueError("spec needs both axes bounded")
        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be finite and > 0, got {self.dt!r}")


def su_shapes(geom: ScalarGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    m, c = (nz, ny, nx), (ny, nx)
    return {"q": m, "tend": m, "q0": m, "held": m, "ru_m": (nz, ny, nx + 1),
            "rv_m": (nz, ny + 1, nx), "ww_m": (nz + 1, ny, nx), "mup0": c, "mup": c,
            "mub2d": c, "msft": c, "c1h": (nz,), "c2h": (nz,), "rdnw": (nz,),
            "fnm": (nz,), "fnp": (nz,)}


def su_active(geom: ScalarGeometry) -> tuple:
    return SU_ACTIVE if geom.held else tuple(n for n in SU_ACTIVE if n != "held")


def _su_check(geom, named) -> np.dtype:
    """Shapes, contiguity, distinct buffers and one real type over every
    array the call names (the species is named once: in the step it is
    one array in the state and the trajectory, and in the perturbations)."""
    geom.validate()
    _check(geom, named, su_shapes(geom), min_nz=1)
    return _dtype([a for _, _, a in named])


_ADV_NAMES = {"tend": "tend", "q": "q", "ru": "ru_m", "rv": "rv_m", "rw": "ww_m",
              "msf": "msft", "rdz": "rdnw", "fnm": "fnm", "fnp": "fnp"}


def _adv_view(d: dict) -> dict:
    """The update's names as flux_div_scalar takes them (missing names
    left out)."""
    return {n: d[m] for n, m in _ADV_NAMES.items() if m in d}


def _ring(a, sz: int) -> None:
    """Zero the specified ring (moist._exclude_specified_ring_advection)."""
    a[:, :sz, :] = 0
    a[:, -sz:, :] = 0
    a[:, sz:-sz, :sz] = 0
    a[:, sz:-sz, -sz:] = 0


def _masses(traj, dtype):
    c1h = traj["c1h"][:, None, None]
    c2h = traj["c2h"][:, None, None]
    mu0 = traj["mub2d"] + traj["mup0"]
    mu = traj["mub2d"] + traj["mup"]
    chm0 = c1h * mu0[None] + c2h
    chm = c1h * mu[None] + c2h
    return c1h, chm0, chm


def _tendency(tend, held, msft, geom: ScalarGeometry) -> None:
    """The advective tendency to the update's (the strict
    rk_update_scalar's ``advect_tend*msfty + sc_tend``, the advection only
    inside the specified ring): ring cleared, times msft, plus the held
    boundary tendency."""
    if geom.spec:
        _ring(tend, geom.spec_zone)
    if geom.has_msf:
        tend *= msft[None]
    if geom.held:
        tend += held


def _base_tendency(traj, geom: ScalarGeometry):
    """The update's tendency on ``traj`` (the adjoint's recompute), through
    the base-only advection build."""
    import cupy as cp
    from ._mode import base_only
    tend = cp.zeros_like(traj["q"])
    work = {**traj, "tend": tend}
    scratch = {n: cp.zeros_like(work[n]) for n in ("tend", "q", "ru_m", "rv_m", "ww_m")}
    with base_only():
        adv_tl(_adv_view(work), _adv_view(scratch), _adv_view(work),
               _adv_view(scratch), geom.adv(), nl=True)
    _tendency(tend, traj.get("held"), traj["msft"], geom)
    return tend


def su_tl(state, pert_state, traj, pert, geom: ScalarGeometry, *, nl=True) -> None:
    """TL of the species' stage update: ``state['q']`` (and the scratch
    ``tend``) from the trajectory, ``pert_state['q']`` from the
    perturbations of :func:`su_active`."""
    tl = _tangent()
    names = [("state", n, state[n]) for n in SU_STATE]
    names += [("traj", n, traj[n]) for n in su_active(geom) + SU_PASSIVE
              if n not in SU_STATE]
    if tl:
        names += [("pert", n, pert_state[n]) for n in SU_STATE]
        names += [("pert", n, pert[n]) for n in su_active(geom) if n not in SU_STATE]
    dtype = _su_check(geom, names)
    dt = dtype.type(geom.dt)
    if nl:
        state["tend"][...] = 0
    if tl:
        pert_state["tend"][...] = 0
    adv_tl(_adv_view(state), _adv_view(pert_state), _adv_view(traj),
           _adv_view(pert), geom.adv(), nl=nl)
    if nl:
        tend = state["tend"]
        _tendency(tend, traj.get("held"), traj["msft"], geom)
    else:
        tend = _base_tendency(traj, geom)
    c1h, chm0, chm = _masses(traj, dtype)
    if tl:
        _tendency(pert_state["tend"], pert.get("held"), traj["msft"], geom)
        dv = (c1h * pert["mup0"][None]) * traj["q0"]
        dv += chm0 * pert["q0"]
        dv += dt * pert_state["tend"]
    q = chm0 * traj["q0"]
    q += dt * tend
    q /= chm
    if tl:
        pert_state["q"][...] = (dv - q * (c1h * pert["mup"][None])) / chm
    if nl:
        state["q"][...] = q


def su_ad(adj_state, traj, adj, geom: ScalarGeometry) -> None:
    """Adjoint of :func:`su_tl`: ``adj_state['q']`` (the updated
    species' adjoint) into ``adj`` (:func:`su_active`); ``adj['q']`` is
    the adjoint of the stage's species the advection read, so the
    update's own seed is consumed first.  ``traj`` is the base before
    the update."""
    names = [("adj", n, adj_state[n]) for n in SU_STATE]
    names += [("traj", n, traj[n]) for n in su_active(geom) + SU_PASSIVE]
    names += [("adj", n, adj[n]) for n in su_active(geom) if n not in SU_STATE]
    dtype = _su_check(geom, names)
    dt = dtype.type(geom.dt)
    c1h, chm0, chm = _masses(traj, dtype)
    q = chm0 * traj["q0"]
    q += dt * _base_tendency(traj, geom)
    q /= chm                                   # the updated species
    r = adj_state["q"] / chm
    adj_state["q"][...] = 0
    adj["q0"] += chm0 * r
    adj["mup0"] += (c1h * traj["q0"] * r).sum(axis=0)
    adj["mup"] -= (c1h * q * r).sum(axis=0)
    a = dt * r
    if geom.held:
        adj["held"] += a
    if geom.has_msf:
        a *= traj["msft"][None]
    if geom.spec:
        _ring(a, geom.spec_zone)
    adj_state["tend"][...] = a
    adv_ad(_adv_view(adj_state), _adv_view(traj), _adv_view(adj), geom.adv())
    adj_state["tend"][...] = 0
