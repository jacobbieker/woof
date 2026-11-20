"""RK-stage glue TL/AD (P3a item 4): calc_p_rho_phi, rk_step_prep
(couple_momentum + calc_ww_cp), rk_addtend_dry and set_w_surface, on
woof's strict primal (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`: dicts of arrays by
name, ``*_tl(state, pert_state, traj, pert, geom, nl=True)`` updates the
perturbation (and the base state when ``nl``), ``*_ad`` maps written
outputs' adjoints to zero and adds into the inputs' adjoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._mode import tl_options
from .acoustic import AD_KCHUNK, THREADS, _blocks, _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"
LIBM = Path(__file__).resolve().parents[2] / "core" / "kernels" / "glibc_flt32.cuh"


@lru_cache(maxsize=None)
def _device_module(real: str, device: int, build: tuple = ()):
    import cupy as cp
    source = (KERNELS / "rk_stage.cu").read_text(encoding="utf-8")
    if real == "float":
        # The strict diagnosis's binary32 log/pow (MIT-licensed Arm
        # optimized-routines transcriptions), so the float base state
        # gives the primal's words.
        source = (LIBM.read_text(encoding="utf-8")
                  + "\n#define WRFPLUS_FLOAT_LIBM 1\n" + source)
    return cp.RawModule(code=source, options=(
        "--fmad=false", f"-DWRFPLUS_REAL={real}", f"-DRK_KCHUNK={AD_KCHUNK}") + build)


def _kernel(name: str, dtype):
    import cupy as cp
    return _device_module(_real(dtype), cp.cuda.Device().id, tl_options()).get_function(name)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


# ------------------------------------------------------------ calc_p_rho_phi

PRP_STATE = ("p", "al", "alt", "p_pp")
PRP_ACTIVE = ("thp", "php", "mup", "muts")
PRP_PASSIVE = ("mub", "phb", "alb", "pb", "rdnw", "c1h", "c2h", "c3h", "c4h",
               "c3f", "c4f")


@dataclass(frozen=True)
class PRPGeometry:
    """calc_p_alpha's scalars.  ``carried`` reads WRF's grid%muts from
    ``muts`` (calc_p_alpha_carried_muts); otherwise muts = mub + mup and
    ``muts`` is not read."""
    nz: int
    ny: int
    nx: int
    hypso: int = 1
    base3d: int = 1
    carried: int = 1
    p_top: float = 0.0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.hypso),
                i(self.base3d), i(self.carried), r(self.p_top))


def prp_shapes(geom: PRPGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    b3 = geom.base3d
    return {"p": mass, "al": mass, "alt": mass, "p_pp": mass, "thp": mass,
            "php": wlev, "mup": col, "muts": col, "mub": col,
            "phb": wlev if b3 else (nz + 1,), "alb": mass if b3 else (nz,),
            "pb": mass if b3 else (nz,), "rdnw": (nz,), "c1h": (nz,),
            "c2h": (nz,), "c3h": (nz,), "c4h": (nz,), "c3f": (nz + 1,),
            "c4f": (nz + 1,)}


def calc_p_rho_phi_tl(state, pert_state, traj, pert, geom, *, nl=True):
    """WRFPLUS G_CALC_P_RHO_PHI (dry, non-hydrostatic) on woof's strict
    calc_p_alpha[_carried_muts]: p, al, alt and p_pp from thp, php, mup
    (and muts when carried)."""
    sh = prp_shapes(geom)
    _check(geom, _named("state", PRP_STATE, state)
           + _named("pert", PRP_STATE, pert_state)
           + _named("traj", PRP_ACTIVE + PRP_PASSIVE, traj)
           + _named("pert", PRP_ACTIVE, pert), sh, min_nz=1)
    arrs = (tuple(state[n] for n in PRP_STATE)
            + tuple(pert_state[n] for n in PRP_STATE)
            + tuple(traj[n] for n in PRP_ACTIVE)
            + tuple(pert[n] for n in PRP_ACTIVE)
            + tuple(traj[n] for n in PRP_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("calc_p_rho_phi_tl", dtype)(
        _blocks(geom.ny * geom.nx), (THREADS,),
        arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def calc_p_rho_phi_ad(adj_state, traj, adj, geom):
    sh = prp_shapes(geom)
    _check(geom, _named("adj", PRP_STATE, adj_state)
           + _named("traj", PRP_ACTIVE + PRP_PASSIVE, traj)
           + _named("adj", PRP_ACTIVE, adj), sh, min_nz=1)
    arrs = (tuple(adj_state[n] for n in PRP_STATE)
            + tuple(traj[n] for n in PRP_ACTIVE)
            + tuple(adj[n] for n in PRP_ACTIVE)
            + tuple(traj[n] for n in PRP_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("calc_p_rho_phi_ad", dtype)(
        _blocks(geom.ny * geom.nx), (THREADS,), arrs + geom.args(dtype))


# -------------------------------------------------------------- rk_step_prep

RSP_STATE = ("ru", "rv", "ww")
RSP_ACTIVE = ("u", "v", "mup")
RSP_PASSIVE = ("mub", "c1h", "c2h", "dnw", "msfu", "msfv", "msft")


@dataclass(frozen=True)
class RSPGeometry:
    """stage_fluxes' scalars.  ``msfv`` is the strict path's msfv_inv (a
    multiplier); the map factors are read only when ``has_msf``.
    ``rdx``/``rdy`` are 1/dx, 1/dy as the primal rounds them."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    has_msf: int = 0
    boundary_x: int = 0
    boundary_y: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.has_msf),
                i(self.boundary_x), i(self.boundary_y), r(self.rdx),
                r(self.rdy))


def rsp_shapes(geom: RSPGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    us, vs, col = (nz, ny, nx + 1), (nz, ny + 1, nx), (ny, nx)
    return {"ru": us, "rv": vs, "ww": (nz + 1, ny, nx), "u": us, "v": vs,
            "mup": col, "mub": col, "c1h": (nz,), "c2h": (nz,),
            "dnw": (nz,), "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
            "msft": col}


def rk_step_prep_tl(state, pert_state, traj, pert, geom, *, nl=True):
    """WRFPLUS g_calc_mu_uv + g_couple_momentum + g_calc_ww_cp (the dry
    part of g_rk_step_prep) on woof's strict stage_fluxes: ru, rv, ww.

    Boundary face masses are clamped or wrapped; the rk_step 1
    ``carried_wrf_mu_halo`` override of a non-periodic axis
    (``stage_fluxes(first_stage=True)``) is not covered (item 7)."""
    sh = rsp_shapes(geom)
    _check(geom, _named("state", RSP_STATE, state)
           + _named("pert", RSP_STATE, pert_state)
           + _named("traj", RSP_ACTIVE + RSP_PASSIVE, traj)
           + _named("pert", RSP_ACTIVE, pert), sh, min_nz=1)
    arrs = (tuple(state[n] for n in RSP_STATE)
            + tuple(pert_state[n] for n in RSP_STATE)
            + tuple(traj[n] for n in RSP_ACTIVE)
            + tuple(pert[n] for n in RSP_ACTIVE)
            + tuple(traj[n] for n in RSP_PASSIVE))
    dtype = _dtype(arrs)
    g = geom
    tail = (np.int32(bool(nl)),) + g.args(dtype)
    _kernel("rk_step_prep_flux_tl", dtype)(
        _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), (THREADS,), arrs + tail)
    _kernel("rk_step_prep_ww_tl", dtype)(
        _blocks(g.ny * g.nx), (THREADS,), arrs + tail)


def rk_step_prep_ad(adj_state, traj, adj, geom):
    import cupy as cp
    sh = rsp_shapes(geom)
    _check(geom, _named("adj", RSP_STATE, adj_state)
           + _named("traj", RSP_ACTIVE + RSP_PASSIVE, traj)
           + _named("adj", RSP_ACTIVE, adj), sh, min_nz=1)
    sa = tuple(adj_state[n] for n in RSP_STATE)
    tr = tuple(traj[n] for n in RSP_ACTIVE)
    ps = tuple(traj[n] for n in RSP_PASSIVE)
    dtype = _dtype(sa + tr + ps + tuple(adj[n] for n in RSP_ACTIVE))
    g = geom
    nck = (g.nz + AD_KCHUNK - 1) // AD_KCHUNK
    bk = cp.empty((g.nz, g.ny, g.nx), dtype=dtype)
    xm = cp.empty((nck, g.ny, g.nx + 1), dtype=dtype)
    ym = cp.empty((nck, g.ny + 1, g.nx), dtype=dtype)
    geo = g.args(dtype)
    _kernel("rk_step_prep_ww_ad", dtype)(
        _blocks(g.ny * g.nx), (THREADS,), sa + (bk,) + ps + geo)
    _kernel("rk_step_prep_flux_ad", dtype)(
        _blocks(nck * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        sa + (bk,) + tr + (adj["u"], adj["v"], xm, ym) + ps + geo)
    _kernel("rk_step_prep_mass_gather", dtype)(
        _blocks(g.ny * g.nx), (THREADS,), (adj["mup"], xm, ym) + geo)


# ------------------------------------------------------------ rk_addtend_dry

AT_STATE = ("ru_t", "rv_t", "rw_t", "rph_t", "rth_t", "rmu_t")
AT_ACTIVE = ("ru_tf", "rv_tf", "rw_tf", "rph_tf", "rth_tf", "rmu_tf",
             "h_diabatic", "mup")
AT_PASSIVE = ("mub", "c1h", "c2h", "msft")


@dataclass(frozen=True)
class ATGeometry:
    """``heating`` adds the held microphysics heating (mp_physics != 0)."""
    nz: int
    ny: int
    nx: int
    has_msf: int = 0
    heating: int = 1

    def args(self, dtype) -> tuple:
        i = np.int32
        return (i(self.nz), i(self.ny), i(self.nx), i(self.has_msf),
                i(self.heating))


def at_shapes(geom: ATGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    us, vs = (nz, ny, nx + 1), (nz, ny + 1, nx)
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    out = {}
    for n, s in (("ru", us), ("rv", vs), ("rw", wlev), ("rph", wlev),
                 ("rth", mass), ("rmu", col)):
        out[n + "_t"] = s
        out[n + "_tf"] = s
    out.update({"h_diabatic": mass, "mup": col, "mub": col, "c1h": (nz,),
                "c2h": (nz,), "msft": col})
    return out


def _at_size(geom) -> int:
    return max((geom.nz + 1) * (geom.ny + 1) * (geom.nx + 1), 1)


def rk_addtend_dry_tl(state, pert_state, traj, pert, geom, *, nl=True):
    """WRFPLUS G_RK_ADDTEND_DRY (rk_step > 1: no relaxation fold) on
    woof's add_fixed_dry_tendencies + add_held_heating: the held tendf
    here are woof's, already divided by the map factor."""
    sh = at_shapes(geom)
    _check(geom, _named("state", AT_STATE, state)
           + _named("pert", AT_STATE, pert_state)
           + _named("traj", AT_ACTIVE + AT_PASSIVE, traj)
           + _named("pert", AT_ACTIVE, pert), sh, min_nz=1)
    arrs = (tuple(state[n] for n in AT_STATE)
            + tuple(pert_state[n] for n in AT_STATE)
            + tuple(traj[n] for n in AT_ACTIVE)
            + tuple(pert[n] for n in AT_ACTIVE)
            + tuple(traj[n] for n in AT_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("rk_addtend_dry_tl", dtype)(
        _blocks(_at_size(geom)), (THREADS,),
        arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def rk_addtend_dry_ad(adj_state, traj, adj, geom):
    sh = at_shapes(geom)
    _check(geom, _named("adj", AT_STATE, adj_state)
           + _named("traj", AT_ACTIVE + AT_PASSIVE, traj)
           + _named("adj", AT_ACTIVE, adj), sh, min_nz=1)
    arrs = (tuple(adj_state[n] for n in AT_STATE)
            + tuple(traj[n] for n in AT_ACTIVE)
            + tuple(adj[n] for n in AT_ACTIVE)
            + tuple(traj[n] for n in AT_PASSIVE))
    dtype = _dtype(arrs)
    geo = geom.args(dtype)
    _kernel("rk_addtend_dry_ad", dtype)(
        _blocks(_at_size(geom)), (THREADS,), arrs + geo)
    _kernel("rk_addtend_dry_ad_mass", dtype)(
        _blocks(geom.ny * geom.nx), (THREADS,), arrs + geo)


# ------------------------------------------------------------- set_w_surface

SW_ACTIVE = ("u", "v")
SW_PASSIVE = ("ht", "msft")


@dataclass(frozen=True)
class SWGeometry:
    """set_surface_w's scalars; ``half_rdx`` = 0.5/dx as the primal
    rounds it."""
    nz: int
    ny: int
    nx: int
    cf1: float
    cf2: float
    cf3: float
    half_rdx: float
    half_rdy: float
    mapped: int = 0
    boundary_x: int = 0
    boundary_y: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.mapped),
                i(self.boundary_x), i(self.boundary_y), r(self.cf1),
                r(self.cf2), r(self.cf3), r(self.half_rdx), r(self.half_rdy))


def sw_shapes(geom: SWGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    return {"w": (nz + 1, ny, nx), "u": (nz, ny, nx + 1),
            "v": (nz, ny + 1, nx), "ht": (ny, nx), "msft": (ny, nx)}


def set_w_surface_tl(state, pert_state, traj, pert, geom, *, nl=True):
    """WRFPLUS g_set_w_surface (fill_w_flag false) on woof's strict
    set_surface_w: w at the surface from u, v levels 0-2."""
    sh = sw_shapes(geom)
    _check(geom, [("state", "w", state["w"]), ("pert", "w", pert_state["w"])]
           + _named("traj", SW_ACTIVE + SW_PASSIVE, traj)
           + _named("pert", SW_ACTIVE, pert), sh)
    arrs = (state["w"], pert_state["w"], traj["u"], traj["v"], pert["u"],
            pert["v"], traj["ht"], traj["msft"])
    dtype = _dtype(arrs)
    _kernel("set_w_surface_tl", dtype)(
        _blocks(geom.ny * geom.nx), (THREADS,),
        arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def set_w_surface_ad(adj_state, traj, adj, geom):
    sh = sw_shapes(geom)
    _check(geom, [("adj", "w", adj_state["w"])]
           + _named("traj", SW_PASSIVE, traj)
           + _named("adj", SW_ACTIVE, adj), sh)
    arrs = (adj_state["w"], adj["u"], adj["v"], traj["ht"], traj["msft"])
    dtype = _dtype(arrs)
    g = geom
    _kernel("set_w_surface_ad", dtype)(
        _blocks((g.ny + 1) * (g.nx + 1)), (THREADS,), arrs + g.args(dtype))
    _kernel("set_w_surface_ad_consume", dtype)(
        _blocks(g.ny * g.nx), (THREADS,),
        (adj_state["w"], np.int32(g.ny * g.nx)))


# ------------------------------------------------------ one RK stage, chained

#: The glue operators' names on the stage's shared dict: calc_p_rho_phi's
#: p_pp is woof's p_perturbation (not the acoustic p''), rk_step_prep's
#: ww is advance_mu_t's stage Omega, mub is mub2d everywhere.
PRP_ALIAS = {"p_pp": "p_pert", "mub": "mub2d"}
RSP_ALIAS = {"ww": "ww_ref", "mub": "mub2d"}
AT_ALIAS = {"mub": "mub2d"}


def _view(d: dict, alias: dict) -> dict:
    """``d`` seen under an operator's own names (missing names left out)."""
    out = {n: v for n, v in d.items() if n not in alias.values()}
    out.update({n: d[m] for n, m in alias.items() if m in d})
    return out


@dataclass(frozen=True)
class StageGeometry:
    """One RK stage in woof's strict order with the big-step tendencies
    held (item 4's chained test; item 8's driver composes the slow
    tendencies in).  ``ss.rk_step`` picks small_step_prep's MUTS."""
    prp: PRPGeometry
    rsp: RSPGeometry
    at: ATGeometry
    ss: object          # acoustic.SSGeometry
    cw: object          # acoustic.CWGeometry
    substep: tuple      # (UVGeometry, MTGeometry, WGeometry)
    nsub: int = 2


def _stage_written():
    from .acoustic import CW_STATE, SF_STATE, SI_STATE
    return {"prp": ("p", "al", "alt", "p_pert"), "rsp": ("ru", "rv", "ww_ref"),
            "at": AT_STATE, "si": SI_STATE, "muts": ("muts",), "cw": CW_STATE,
            "sf": SF_STATE}


def rk_stage_tl(state: dict, pert: dict, geom: StageGeometry,
                tape: list | None = None) -> None:
    """One RK stage with the big-step tendencies held: calc_p_rho_phi
    (carried muts), rk_step_prep, rk_addtend_dry, small_step_init,
    small_step_prep's MUTS, calc_coef_w, ``nsub`` acoustic substeps,
    the captured MUTS = MUB + MU + MU'' and small_step_finish, as
    woof/core/dycore.py's strict stage runs them.  ``state`` advances
    and ``pert`` maps in place; with a ``tape`` each operator's written
    state is appended before it runs, for :func:`rk_stage_ad`."""
    import cupy as cp
    from .acoustic import (acoustic_substep_tl, calc_coef_w_tl,
                           small_step_finish_tl, small_step_init_tl)
    W = _stage_written()
    snap = (lambda k: tape.append({n: state[n].copy() for n in W[k]})) \
        if tape is not None else (lambda k: None)
    for k, (fn, alias, g) in (("prp", (calc_p_rho_phi_tl, PRP_ALIAS, geom.prp)),
                              ("rsp", (rk_step_prep_tl, RSP_ALIAS, geom.rsp)),
                              ("at", (rk_addtend_dry_tl, AT_ALIAS, geom.at))):
        snap(k)
        s, p = _view(state, alias), _view(pert, alias)
        fn(s, p, s, p, g, nl=True)
    snap("si")
    small_step_init_tl(state, pert, state, pert, geom.ss, nl=True)
    snap("muts")
    src = "mup" if geom.ss.rk_step == 1 else "mup0"
    cp.add(state["mub2d"], state[src], out=state["muts"])
    pert["muts"][...] = pert[src]
    snap("cw")
    calc_coef_w_tl(state, pert, state, pert, geom.cw, nl=True)
    for _ in range(geom.nsub):
        acoustic_substep_tl(state, pert, geom.substep, tape)
    snap("muts")
    cp.add(cp.add(state["mub2d"], state["mup"]), state["mu_pp"], out=state["muts"])
    cp.add(pert["mup"], pert["mu_pp"], out=pert["muts"])
    snap("sf")
    small_step_finish_tl(state, pert, state, pert, geom.ss, nl=True)


def rk_stage_ad(adj: dict, state: dict, tape: list, geom: StageGeometry) -> dict:
    """Adjoint of :func:`rk_stage_tl`: ``adj`` (every name the stage
    reads or writes that is perturbed) maps from the output adjoint to
    the input adjoint in place; ``state`` is the base after the stage and
    ``tape`` what its TL appended.  Returns the base before the stage."""
    from .acoustic import (SF_STATE, acoustic_substep_ad, calc_coef_w_ad,
                           small_step_finish_ad, small_step_init_ad)
    cur = dict(state)

    def back():
        nonlocal cur
        cur = {**cur, **tape.pop()}
        return cur

    base = back()                                   # sf
    small_step_finish_ad(adj, {n: base[n] for n in SF_STATE}, base, adj, geom.ss)
    back()                                          # captured muts
    adj["mup"] += adj["muts"]
    adj["mu_pp"] += adj["muts"]
    adj["muts"][...] = 0
    for _ in range(geom.nsub):
        cur = acoustic_substep_ad(adj, cur, tape, geom.substep)
    base = back()                                   # cw
    calc_coef_w_ad(adj, base, adj, geom.cw)
    back()                                          # small_step_prep's muts
    src = "mup" if geom.ss.rk_step == 1 else "mup0"
    adj[src] += adj["muts"]
    adj["muts"][...] = 0
    base = back()                                   # si
    small_step_init_ad(adj, base, adj, geom.ss)
    for fn, alias, g in ((rk_addtend_dry_ad, AT_ALIAS, geom.at),
                         (rk_step_prep_ad, RSP_ALIAS, geom.rsp),
                         (calc_p_rho_phi_ad, PRP_ALIAS, geom.prp)):
        base = back()
        a = _view(adj, alias)
        fn(a, _view(base, alias), a, g)
    return cur
